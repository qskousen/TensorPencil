/* Minimal libvips shim for tp-llm's image DECODE (--image / @mentions),
   adapted from DiffKeep's lib/vips/vips_helper.c. libvips's C API is
   varargs-based (NULL-terminated option lists), which @cImport cannot call
   directly — this wrapper flattens the two entry points we need.

   Linked into the tp-llm EXECUTABLE only; the TensorPencil library module
   stays pure Zig (see CLAUDE.md "Dependencies"). */
#pragma once
#include <stddef.h>

int tp_vips_init(const char *name);
void tp_vips_shutdown(void);
void tp_vips_free(void *buf);

/* Load an image file (any libvips-supported format: jpeg, png, webp, gif,
   tiff, ...) as packed interleaved RGB8 at its native resolution,
   EXIF-autorotated, alpha flattened over white. Returns 0 on success; *buf
   is g_malloc-allocated (free with tp_vips_free). */
int tp_load_image_rgb(
    const char *filename,
    void **buf,
    size_t *len,
    int *width,
    int *height
);
