#pragma once
#include <stddef.h>
#include <stdint.h>

/* Thin C wrapper over libav* for TensorPencil's clip MUXING. Portable across
   Linux/macOS/Windows; only build.zig wires the platform include/lib paths.
   The DECODE side lives in DiffKeep (lib/video/video_helper.c) and is a
   different concern; this file only writes. */

typedef struct {
    int width;
    int height;
    int fps_num;
    int fps_den;
    /* 1..51, lower is better quality. 0 selects the encoder default. */
    int crf;
    /* 0 disables the audio stream entirely. */
    int audio_channels;
    int audio_sample_rate;
    /* Format-level metadata written into the container (the AUTOMATIC1111
       `parameters` block, so a reader can re-render from the file). NULL to
       omit. `meta_key` is typically "parameters". */
    const char *meta_key;
    const char *meta_value;
} TpMuxConfig;

typedef struct TpMuxer TpMuxer;

/* Open `filename` for writing. NULL on failure. The container is chosen from
   the extension. */
TpMuxer *tp_mux_open(const char *filename, const TpMuxConfig *cfg);

/* Append one frame of packed RGB8 (width*height*3 bytes). 0 on success. */
int tp_mux_write_frame(TpMuxer *m, const uint8_t *rgb);

/* Append interleaved float32 audio in [-1, 1], `n_frames` per channel.
   May be called once with the whole track. 0 on success. */
int tp_mux_write_audio(TpMuxer *m, const float *interleaved, size_t n_frames);

/* Flush both encoders, write the trailer and close. 0 on success. Frees `m`
   either way. */
int tp_mux_finish(TpMuxer *m);

/* Abort without finishing the file (frees `m`). */
void tp_mux_abort(TpMuxer *m);

/* Last error text for the calling thread, or NULL. Valid until the next call. */
const char *tp_mux_last_error(void);
