#include "vips_helper.h"
#include <vips/vips.h>

int tp_vips_init(const char *name) {
    int rc = vips_init(name);
    if (rc == 0) {
        /* One-shot decodes only: vips's operation cache buys nothing here
           and costs memory (same reasoning as DiffKeep's thumbnailer). */
        vips_cache_set_max(0);
        vips_cache_set_max_mem(0);
        vips_cache_set_max_files(0);
    }
    return rc;
}

void tp_vips_shutdown(void) {
    vips_shutdown();
}

void tp_vips_free(void *buf) {
    g_free(buf);
}

int tp_load_image_rgb(
    const char *filename,
    void **buf,
    size_t *len,
    int *width,
    int *height
) {
    /* vips_thumbnail with a huge target and VIPS_SIZE_DOWN returns the image
       at its original resolution converted to sRGB(A) uint8 (DiffKeep's
       proven full-res load path). EXIF autorotation is the default. */
    VipsImage *image = NULL;
    if (vips_thumbnail(filename, &image, 65535,
            "height", 65535,
            "size", VIPS_SIZE_DOWN,
            NULL) != 0)
        return -1;

    /* Vision input wants exactly 3-band sRGB: flatten alpha over white
       (screenshots/diagrams with transparency read as on-paper). */
    if (vips_image_hasalpha(image)) {
        VipsImage *flat = NULL;
        VipsArrayDouble *bg = vips_array_double_newv(3, 255.0, 255.0, 255.0);
        int rc = vips_flatten(image, &flat, "background", bg, NULL);
        vips_area_unref(VIPS_AREA(bg));
        if (rc != 0) {
            g_object_unref(image);
            return -1;
        }
        g_object_unref(image);
        image = flat;
    }
    if (vips_image_get_bands(image) != 3 ||
        vips_image_get_format(image) != VIPS_FORMAT_UCHAR) {
        VipsImage *srgb = NULL;
        if (vips_colourspace(image, &srgb, VIPS_INTERPRETATION_sRGB, NULL) != 0) {
            g_object_unref(image);
            return -1;
        }
        g_object_unref(image);
        image = srgb;
        if (vips_image_get_bands(image) != 3 ||
            vips_image_get_format(image) != VIPS_FORMAT_UCHAR) {
            g_object_unref(image);
            return -1;
        }
    }

    size_t out_len = 0;
    void *pixels = vips_image_write_to_memory(image, &out_len);
    if (!pixels) {
        g_object_unref(image);
        return -1;
    }

    *width = vips_image_get_width(image);
    *height = vips_image_get_height(image);
    *buf = pixels;
    *len = out_len;
    g_object_unref(image);
    return 0;
}
