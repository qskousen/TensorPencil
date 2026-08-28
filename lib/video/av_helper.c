/* libav* muxer for TensorPencil clips: H.264 video plus optional AAC audio.
 *
 * Deliberately narrow. It writes one video stream, at most one audio stream,
 * and one format-level metadata tag, because that is what a rendered clip is.
 * Anything more general belongs in a real media layer.
 *
 * Two things here are easy to get wrong and silent when wrong:
 *
 * 1. Every timestamp is in its STREAM's time base, not the codec's. The pattern
 *    is encode in the codec time base, then `av_packet_rescale_ts` on the way
 *    out. Skipping that gives a file whose duration is wrong by the ratio of the
 *    two bases, which players show as absurd playback speed rather than an error.
 * 2. The AAC encoder wants a FIXED frame size and float PLANAR samples. Handing
 *    it interleaved data reads channel 1 as the tail of channel 0, which sounds
 *    like a click track rather than silence.
 */
#include "av_helper.h"

#include <stdio.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>

static _Thread_local char g_err[512];

static void set_err(const char *what, int rc) {
    char buf[256] = {0};
    if (rc != 0) av_strerror(rc, buf, sizeof buf);
    snprintf(g_err, sizeof g_err, "%s%s%s", what, rc ? ": " : "", rc ? buf : "");
}

const char *tp_mux_last_error(void) { return g_err[0] ? g_err : NULL; }

struct TpMuxer {
    AVFormatContext *fc;

    AVStream *vs;
    AVCodecContext *vc;
    AVFrame *vframe;
    struct SwsContext *sws;
    int64_t vpts;

    AVStream *as;
    AVCodecContext *ac;
    AVFrame *aframe;
    SwrContext *swr;
    int64_t apts;
    /* Partial AAC frame carried between `tp_mux_write_audio` calls. */
    float *pending;      /* interleaved */
    int pending_frames;

    AVPacket *pkt;
    int header_written;
};

static int drain(TpMuxer *m, AVCodecContext *cc, AVStream *st) {
    for (;;) {
        int rc = avcodec_receive_packet(cc, m->pkt);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return 0;
        if (rc < 0) { set_err("avcodec_receive_packet", rc); return -1; }
        /* Codec time base -> stream time base. See the header note. */
        av_packet_rescale_ts(m->pkt, cc->time_base, st->time_base);
        m->pkt->stream_index = st->index;
        rc = av_interleaved_write_frame(m->fc, m->pkt);
        av_packet_unref(m->pkt);
        if (rc < 0) { set_err("av_interleaved_write_frame", rc); return -1; }
    }
}

static void destroy(TpMuxer *m) {
    if (!m) return;
    if (m->sws) sws_freeContext(m->sws);
    if (m->swr) swr_free(&m->swr);
    if (m->vframe) av_frame_free(&m->vframe);
    if (m->aframe) av_frame_free(&m->aframe);
    if (m->vc) avcodec_free_context(&m->vc);
    if (m->ac) avcodec_free_context(&m->ac);
    if (m->pkt) av_packet_free(&m->pkt);
    if (m->fc) {
        if (m->fc->pb) avio_closep(&m->fc->pb);
        avformat_free_context(m->fc);
    }
    free(m->pending);
    free(m);
}

TpMuxer *tp_mux_open(const char *filename, const TpMuxConfig *cfg) {
    g_err[0] = 0;
    if (!filename || !cfg || cfg->width <= 0 || cfg->height <= 0) {
        set_err("bad arguments", 0);
        return NULL;
    }
    /* x264 and the AAC encoder print per-file statistics at INFO; a render is
       not the place for them. Errors still come through. */
    av_log_set_level(AV_LOG_ERROR);
    TpMuxer *m = calloc(1, sizeof *m);
    if (!m) { set_err("out of memory", 0); return NULL; }
    m->pkt = av_packet_alloc();
    if (!m->pkt) { set_err("av_packet_alloc", 0); goto fail; }

    int rc = avformat_alloc_output_context2(&m->fc, NULL, NULL, filename);
    if (rc < 0 || !m->fc) { set_err("avformat_alloc_output_context2", rc); goto fail; }

    /* ---- video ---------------------------------------------------------- */
    const AVCodec *vcodec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!vcodec) { set_err("no H.264 encoder in this libavcodec", 0); goto fail; }
    m->vs = avformat_new_stream(m->fc, NULL);
    if (!m->vs) { set_err("avformat_new_stream (video)", 0); goto fail; }
    m->vc = avcodec_alloc_context3(vcodec);
    if (!m->vc) { set_err("avcodec_alloc_context3 (video)", 0); goto fail; }

    m->vc->width = cfg->width;
    m->vc->height = cfg->height;
    /* yuv420p: the only pixel format every H.264 decoder is required to take. */
    m->vc->pix_fmt = AV_PIX_FMT_YUV420P;
    m->vc->time_base = (AVRational){cfg->fps_den > 0 ? cfg->fps_den : 1,
                                    cfg->fps_num > 0 ? cfg->fps_num : 24};
    m->vc->framerate = (AVRational){cfg->fps_num > 0 ? cfg->fps_num : 24,
                                    cfg->fps_den > 0 ? cfg->fps_den : 1};
    m->vs->time_base = m->vc->time_base;
    if (m->fc->oformat->flags & AVFMT_GLOBALHEADER)
        m->vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (cfg->crf > 0) {
        char v[8];
        snprintf(v, sizeof v, "%d", cfg->crf);
        av_opt_set(m->vc->priv_data, "crf", v, 0);
    }
    av_opt_set(m->vc->priv_data, "preset", "medium", 0);

    rc = avcodec_open2(m->vc, vcodec, NULL);
    if (rc < 0) { set_err("avcodec_open2 (video)", rc); goto fail; }
    rc = avcodec_parameters_from_context(m->vs->codecpar, m->vc);
    if (rc < 0) { set_err("avcodec_parameters_from_context (video)", rc); goto fail; }

    m->vframe = av_frame_alloc();
    if (!m->vframe) { set_err("av_frame_alloc (video)", 0); goto fail; }
    m->vframe->format = m->vc->pix_fmt;
    m->vframe->width = m->vc->width;
    m->vframe->height = m->vc->height;
    rc = av_frame_get_buffer(m->vframe, 0);
    if (rc < 0) { set_err("av_frame_get_buffer (video)", rc); goto fail; }

    m->sws = sws_getContext(cfg->width, cfg->height, AV_PIX_FMT_RGB24,
                            cfg->width, cfg->height, AV_PIX_FMT_YUV420P,
                            SWS_BILINEAR, NULL, NULL, NULL);
    if (!m->sws) { set_err("sws_getContext", 0); goto fail; }

    /* ---- audio ---------------------------------------------------------- */
    if (cfg->audio_channels > 0 && cfg->audio_sample_rate > 0) {
        const AVCodec *acodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
        if (!acodec) { set_err("no AAC encoder in this libavcodec", 0); goto fail; }
        m->as = avformat_new_stream(m->fc, NULL);
        if (!m->as) { set_err("avformat_new_stream (audio)", 0); goto fail; }
        m->ac = avcodec_alloc_context3(acodec);
        if (!m->ac) { set_err("avcodec_alloc_context3 (audio)", 0); goto fail; }

        m->ac->sample_fmt = AV_SAMPLE_FMT_FLTP; /* AAC wants PLANAR float */
        m->ac->sample_rate = cfg->audio_sample_rate;
        av_channel_layout_default(&m->ac->ch_layout, cfg->audio_channels);
        m->ac->time_base = (AVRational){1, cfg->audio_sample_rate};
        m->as->time_base = m->ac->time_base;
        if (m->fc->oformat->flags & AVFMT_GLOBALHEADER)
            m->ac->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        rc = avcodec_open2(m->ac, acodec, NULL);
        if (rc < 0) { set_err("avcodec_open2 (audio)", rc); goto fail; }
        rc = avcodec_parameters_from_context(m->as->codecpar, m->ac);
        if (rc < 0) { set_err("avcodec_parameters_from_context (audio)", rc); goto fail; }

        m->aframe = av_frame_alloc();
        if (!m->aframe) { set_err("av_frame_alloc (audio)", 0); goto fail; }
        m->aframe->format = m->ac->sample_fmt;
        m->aframe->sample_rate = m->ac->sample_rate;
        av_channel_layout_copy(&m->aframe->ch_layout, &m->ac->ch_layout);
        m->aframe->nb_samples = m->ac->frame_size > 0 ? m->ac->frame_size : 1024;
        rc = av_frame_get_buffer(m->aframe, 0);
        if (rc < 0) { set_err("av_frame_get_buffer (audio)", rc); goto fail; }

        /* Interleaved f32 in -> planar f32 out, same rate and layout. */
        m->swr = swr_alloc();
        if (!m->swr) { set_err("swr_alloc", 0); goto fail; }
        av_opt_set_chlayout(m->swr, "in_chlayout", &m->ac->ch_layout, 0);
        av_opt_set_chlayout(m->swr, "out_chlayout", &m->ac->ch_layout, 0);
        av_opt_set_int(m->swr, "in_sample_rate", cfg->audio_sample_rate, 0);
        av_opt_set_int(m->swr, "out_sample_rate", cfg->audio_sample_rate, 0);
        av_opt_set_sample_fmt(m->swr, "in_sample_fmt", AV_SAMPLE_FMT_FLT, 0);
        av_opt_set_sample_fmt(m->swr, "out_sample_fmt", AV_SAMPLE_FMT_FLTP, 0);
        rc = swr_init(m->swr);
        if (rc < 0) { set_err("swr_init", rc); goto fail; }

        m->pending = calloc((size_t)m->aframe->nb_samples * cfg->audio_channels, sizeof(float));
        if (!m->pending) { set_err("out of memory", 0); goto fail; }
    }

    AVDictionary *mux_opts = NULL;
    if (cfg->meta_key && cfg->meta_value) {
        av_dict_set(&m->fc->metadata, cfg->meta_key, cfg->meta_value, 0);
        /* MP4/MOV keep only the handful of metadata keys they define; an
           arbitrary key is DROPPED SILENTLY without this, which is how the
           `parameters` block vanished the first time. `use_metadata_tags` writes
           unknown keys into a udta atom instead. Unknown to other muxers, which
           just leave the option in the dict. */
        av_dict_set(&mux_opts, "movflags", "use_metadata_tags", 0);
    }

    rc = avio_open(&m->fc->pb, filename, AVIO_FLAG_WRITE);
    if (rc < 0) { av_dict_free(&mux_opts); set_err("avio_open", rc); goto fail; }
    rc = avformat_write_header(m->fc, &mux_opts);
    av_dict_free(&mux_opts);
    if (rc < 0) { set_err("avformat_write_header", rc); goto fail; }
    m->header_written = 1;
    return m;

fail:
    destroy(m);
    return NULL;
}

int tp_mux_write_frame(TpMuxer *m, const uint8_t *rgb) {
    if (!m || !rgb) { set_err("bad arguments", 0); return -1; }
    int rc = av_frame_make_writable(m->vframe);
    if (rc < 0) { set_err("av_frame_make_writable", rc); return -1; }

    const uint8_t *src[4] = {rgb, NULL, NULL, NULL};
    int stride[4] = {m->vc->width * 3, 0, 0, 0};
    sws_scale(m->sws, src, stride, 0, m->vc->height, m->vframe->data, m->vframe->linesize);
    m->vframe->pts = m->vpts++;

    rc = avcodec_send_frame(m->vc, m->vframe);
    if (rc < 0) { set_err("avcodec_send_frame (video)", rc); return -1; }
    return drain(m, m->vc, m->vs);
}

/* Encode exactly one AAC frame from `n` interleaved frames (n <= frame_size).
   A short tail is zero-padded, which is what the encoder's own flush does. */
static int encode_audio_frame(TpMuxer *m, const float *interleaved, int n) {
    int rc = av_frame_make_writable(m->aframe);
    if (rc < 0) { set_err("av_frame_make_writable (audio)", rc); return -1; }
    const int fs = m->aframe->nb_samples;
    const int ch = m->ac->ch_layout.nb_channels;

    const uint8_t *in[1] = {(const uint8_t *)interleaved};
    rc = swr_convert(m->swr, m->aframe->data, fs, in, n);
    if (rc < 0) { set_err("swr_convert", rc); return -1; }
    if (rc < fs) {
        /* Zero the tail so a partial frame does not encode stale samples. */
        for (int c = 0; c < ch; c++)
            memset(m->aframe->data[c] + (size_t)rc * sizeof(float), 0,
                   (size_t)(fs - rc) * sizeof(float));
    }
    m->aframe->pts = m->apts;
    m->apts += fs;
    rc = avcodec_send_frame(m->ac, m->aframe);
    if (rc < 0) { set_err("avcodec_send_frame (audio)", rc); return -1; }
    return drain(m, m->ac, m->as);
}

int tp_mux_write_audio(TpMuxer *m, const float *interleaved, size_t n_frames) {
    if (!m || !m->ac) { set_err("no audio stream", 0); return -1; }
    if (!interleaved && n_frames) { set_err("bad arguments", 0); return -1; }
    const int fs = m->aframe->nb_samples;
    const int ch = m->ac->ch_layout.nb_channels;
    size_t off = 0;

    /* Top up a carried partial frame first. */
    if (m->pending_frames > 0) {
        int want = fs - m->pending_frames;
        int take = (int)(n_frames < (size_t)want ? n_frames : (size_t)want);
        memcpy(m->pending + (size_t)m->pending_frames * ch, interleaved,
               (size_t)take * ch * sizeof(float));
        m->pending_frames += take;
        off += (size_t)take;
        if (m->pending_frames < fs) return 0;
        if (encode_audio_frame(m, m->pending, fs) != 0) return -1;
        m->pending_frames = 0;
    }
    while (n_frames - off >= (size_t)fs) {
        if (encode_audio_frame(m, interleaved + off * ch, fs) != 0) return -1;
        off += (size_t)fs;
    }
    if (off < n_frames) {
        m->pending_frames = (int)(n_frames - off);
        memcpy(m->pending, interleaved + off * ch,
               (size_t)m->pending_frames * ch * sizeof(float));
    }
    return 0;
}

int tp_mux_finish(TpMuxer *m) {
    if (!m) { set_err("bad arguments", 0); return -1; }
    int rc = 0;
    if (m->ac && m->pending_frames > 0) {
        /* Zero-pad and emit the tail rather than dropping it. */
        memset(m->pending + (size_t)m->pending_frames * m->ac->ch_layout.nb_channels, 0,
               (size_t)(m->aframe->nb_samples - m->pending_frames) *
                   m->ac->ch_layout.nb_channels * sizeof(float));
        if (encode_audio_frame(m, m->pending, m->pending_frames) != 0) rc = -1;
        m->pending_frames = 0;
    }
    /* Flush: a null frame tells the encoder to emit whatever it is holding. */
    if (rc == 0 && avcodec_send_frame(m->vc, NULL) >= 0) {
        if (drain(m, m->vc, m->vs) != 0) rc = -1;
    }
    if (rc == 0 && m->ac && avcodec_send_frame(m->ac, NULL) >= 0) {
        if (drain(m, m->ac, m->as) != 0) rc = -1;
    }
    if (rc == 0 && m->header_written) {
        int wr = av_write_trailer(m->fc);
        if (wr < 0) { set_err("av_write_trailer", wr); rc = -1; }
    }
    destroy(m);
    return rc;
}

void tp_mux_abort(TpMuxer *m) { destroy(m); }
