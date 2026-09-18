//! `zig build hosts-probe -- --config <file> --message <text> --image <prompt>
//! [--kill-b] [--llm <file>] [--remote <pairing string>] [--size <px>]
//! [--steps <n>] [--timeout <s>] [--send-model <file>]
//! [--pull-model <stem> --pull-into <dir>]`: tp-gui's multi-host path with no
//! window.
//! Two hosts driven through `client/hosts.zig` exactly as the app drives
//! them: the chat goes to A (the local slot), the image lands on B
//! (`placeRender` keeps it off the chat host). B is a second tp-serve child on
//! a private socket, or with `--remote` the host a daemon's pairing string
//! names, reached under TLS; its catalog must then name models by id only,
//! and the image model is picked from that catalog as the studio would.
//! `--send-model` pushes a file B lacks; `--pull-model` fetches one B has and
//! this machine does not, by the id B named it with, and checks that what
//! landed hashes to the id it was asked for.
//! With `--kill-b` (local B only) B is killed once its render has taken a few
//! steps; the run passes only if A's turn still ends, B's job surfaces failed
//! as lost, and B comes back. Prints one JSON line per state change, exits
//! non-zero on any miss. Both local children share one card, so `--llm` puts
//! a small chat model on A: a 31B beside a diffusion pipeline prefills at one
//! token a second.
const std = @import("std");
const Io = std.Io;
const config = @import("shared").config;
const catalog = @import("shared").catalog;
const wire = @import("serve").wire;
const hosts = @import("client").hosts;
const models = @import("client").models;
const sync = @import("client").sync;
const selection = @import("client").selection;

fn noop() void {}

fn stemOf(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    return base[0 .. std.mem.lastIndexOfScalar(u8, base, '.') orelse base.len];
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = std.heap.smp_allocator;

    const args = try init.minimal.args.toSlice(arena);
    var cfg_path: ?[]const u8 = null;
    var message: ?[]const u8 = null;
    var image: ?[]const u8 = null;
    var kill_b = false;
    var llm: ?[]const u8 = null;
    var pairing: ?[]const u8 = null;
    var send_model: ?[]const u8 = null;
    var pull_model: ?[]const u8 = null;
    var pull_dir: ?[]const u8 = null;
    var size: ?u32 = null;
    var steps: ?u32 = null;
    var timeout_s: u64 = 1200;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const more = i + 1 < args.len;
        if (std.mem.eql(u8, args[i], "--config") and more) {
            i += 1;
            cfg_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--message") and more) {
            i += 1;
            message = args[i];
        } else if (std.mem.eql(u8, args[i], "--image") and more) {
            i += 1;
            image = args[i];
        } else if (std.mem.eql(u8, args[i], "--pull-model") and more) {
            i += 1;
            pull_model = args[i];
        } else if (std.mem.eql(u8, args[i], "--pull-into") and more) {
            i += 1;
            pull_dir = args[i];
        } else if (std.mem.eql(u8, args[i], "--kill-b")) {
            kill_b = true;
        } else if (std.mem.eql(u8, args[i], "--llm") and more) {
            i += 1;
            llm = args[i];
        } else if (std.mem.eql(u8, args[i], "--remote") and more) {
            i += 1;
            pairing = args[i];
        } else if (std.mem.eql(u8, args[i], "--send-model") and more) {
            i += 1;
            send_model = args[i];
        } else if (std.mem.eql(u8, args[i], "--size") and more) {
            i += 1;
            size = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--steps") and more) {
            i += 1;
            steps = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--timeout") and more) {
            i += 1;
            timeout_s = try std.fmt.parseInt(u64, args[i], 10);
        } else {
            std.debug.print("usage: hosts-probe --config <file> --message <text> --image <prompt> [--kill-b] [--llm <file>] [--remote <pairing string>] [--send-model <file>] [--size <px>] [--steps <n>] [--timeout <s>]\n", .{});
            return error.BadArgs;
        }
    }
    const path = cfg_path orelse {
        std.debug.print("hosts-probe: --config is required (never the real settings file)\n", .{});
        return error.BadArgs;
    };
    const msg = message orelse return error.BadArgs;
    const prompt = image orelse return error.BadArgs;
    if (kill_b and pairing != null) {
        std.debug.print("hosts-probe: --kill-b needs a local B\n", .{});
        return error.BadArgs;
    }

    var out_buf: [16 * 1024]u8 = undefined;
    var stdout = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    const out = &stdout.interface;

    var cfg = config.Config.load(io, gpa, init.environ_map, path);
    cfg.applyFraming();
    if (llm) |m| {
        cfg.llm_model.set(m);
        cfg.vision_tower.set("");
    }
    if (size) |px| {
        cfg.width = px;
        cfg.height = px;
    }
    if (steps) |n| cfg.steps = n;
    // Private sockets for both children, so a probe never talks to a host the
    // user is sitting at. B is a listed host: one the client starts itself, or
    // the remote one the pairing string names.
    const stamp = Io.Clock.real.now(io).toSeconds();
    const sock_a = try std.fmt.allocPrint(arena, "/tmp/tp-serve-probe-A-{d}.sock", .{stamp});
    const sock_b = try std.fmt.allocPrint(arena, "/tmp/tp-serve-probe-B-{d}.sock", .{stamp});
    if (!cfg.addHost("B", pairing orelse sock_b, true).ok()) return error.HostListFull;
    const remote_b = pairing != null;

    var h = try hosts.Hosts.init(gpa, io, init.environ_map, noop, path, sock_a);
    defer h.deinit();
    h.sync(&cfg);
    const connect_deadline = Io.Clock.real.now(io).nanoseconds + @as(i96, @intCast(timeout_s)) * std.time.ns_per_s;
    while (h.connecting()) {
        if (Io.Clock.real.now(io).nanoseconds > connect_deadline) return error.ConnectTimeout;
        h.pump(&cfg);
        Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .real) catch {};
    }
    const a = h.local();
    const b = h.slotOf(1) orelse return error.NoHostB;
    if (!b.up()) {
        try out.print("{{\"probe\":\"B down\",\"status\":\"{s}\"}}\n", .{h.statusOf("B")});
        try out.flush();
        return error.HostBDown;
    }
    try out.print("{{\"probe\":\"up\",\"a_gen\":{d},\"b_gen\":{d},\"b_remote\":{}}}\n", .{ a.remote.?.gen, b.remote.?.gen, remote_b });
    try out.flush();

    // Model sync, before anything else: B must end up holding the file under
    // the id this machine computes for it, and say so in its own catalog.
    var syncer = sync.Syncer.init(gpa, io);
    defer syncer.deinit();
    if (send_model) |file| {
        const st = try Io.Dir.cwd().statFile(io, file, .{});
        const id = try catalog.fileId(gpa, io, file);
        const send_deadline = Io.Clock.real.now(io).nanoseconds + @as(i96, @intCast(timeout_s)) * std.time.ns_per_s;
        // B's first catalog has to arrive before "does it have this" means
        // anything.
        while (!b.mirror.scannedOnce()) {
            if (Io.Clock.real.now(io).nanoseconds > send_deadline) return error.CatalogTimeout;
            h.pump(&cfg);
            Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .real) catch {};
        }
        if (b.mirror.catalog.byId(id)) |e| {
            try out.print("{{\"probe\":\"already there\",\"stem\":\"{s}\"}}\n", .{e.stem()});
        } else {
            try syncer.start("B", b.entry.?, file);
            const j = syncer.forHost("B").?;
            try out.print("{{\"probe\":\"sending\",\"file\":\"{s}\",\"bytes\":{d}}}\n", .{ stemOf(file), st.size });
            try out.flush();
            var last_tenth: u32 = std.math.maxInt(u32);
            while (j.running()) {
                if (Io.Clock.real.now(io).nanoseconds > send_deadline) return error.SendTimeout;
                h.pump(&cfg);
                syncer.poll();
                const tenth: u32 = @intFromFloat(j.prog.fraction() * 10);
                if (tenth != last_tenth) {
                    last_tenth = tenth;
                    try out.print("{{\"probe\":\"sending\",\"percent\":{d}}}\n", .{tenth * 10});
                    try out.flush();
                }
                Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .real) catch {};
            }
            syncer.poll();
            var sbuf: [64]u8 = undefined;
            try out.print("{{\"probe\":\"sent\",\"status\":\"{s}\",\"puts\":{d},\"connects\":{d}}}\n", .{
                j.status(&sbuf), j.prog.puts.load(.acquire), j.prog.connects.load(.acquire),
            });
            try out.flush();
            if (j.prog.phase.load(.acquire) != .done) return error.SendFailed;
            // The host rescans itself; the file must come back named by the id
            // this machine computed for it.
            while (b.mirror.catalog.byId(id) == null) {
                if (Io.Clock.real.now(io).nanoseconds > send_deadline) return error.NotInHostCatalog;
                h.pump(&cfg);
                Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .real) catch {};
            }
        }
        const e = b.mirror.catalog.byId(id).?;
        try out.print("{{\"probe\":\"host has it\",\"stem\":\"{s}\",\"ref\":\"{s}\"}}\n", .{ e.stem(), e.path });
        try out.flush();
    }

    // The other direction: a file B has and this machine does not, fetched by
    // the id B named it with. This machine has no path for it, which is the
    // whole reason a pull is not a push run backwards.
    if (pull_model) |stem| {
        const dest = pull_dir orelse return error.NoPullDir;
        const pull_deadline = Io.Clock.real.now(io).nanoseconds + @as(i96, @intCast(timeout_s)) * std.time.ns_per_s;
        while (!b.mirror.scannedOnce()) {
            if (Io.Clock.real.now(io).nanoseconds > pull_deadline) return error.CatalogTimeout;
            h.pump(&cfg);
            Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .real) catch {};
        }
        var want: ?*const catalog.Entry = null;
        for (b.mirror.catalog.entries) |*e| if (std.mem.eql(u8, e.stem(), stem)) {
            want = e;
        };
        const e = want orelse {
            try out.print("{{\"probe\":\"B has no {s}\"}}\n", .{stem});
            try out.flush();
            return error.NotOnHost;
        };
        const id = e.id();
        var id_buf: [catalog.id_text_len]u8 = undefined;
        const id_text = catalog.idText(id, &id_buf);
        try out.print("{{\"probe\":\"pulling\",\"stem\":\"{s}\",\"id\":\"{s}\",\"bytes\":{d}}}\n", .{ stem, id_text, e.size });
        try out.flush();
        try syncer.startPull("B", b.entry.?, id_text, stem, dest);
        const j = syncer.forHost("B").?;
        var last_tenth: u32 = std.math.maxInt(u32);
        while (j.running()) {
            if (Io.Clock.real.now(io).nanoseconds > pull_deadline) return error.PullTimeout;
            h.pump(&cfg);
            syncer.poll();
            const tenth: u32 = @intFromFloat(j.prog.fraction() * 10);
            if (tenth != last_tenth) {
                last_tenth = tenth;
                try out.print("{{\"probe\":\"pulling\",\"percent\":{d}}}\n", .{tenth * 10});
                try out.flush();
            }
            Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .real) catch {};
        }
        syncer.poll();
        var pbuf: [64]u8 = undefined;
        try out.print("{{\"probe\":\"pulled\",\"status\":\"{s}\",\"gets\":{d},\"connects\":{d}}}\n", .{
            j.status(&pbuf), j.prog.puts.load(.acquire), j.prog.connects.load(.acquire),
        });
        try out.flush();
        if (j.prog.phase.load(.acquire) != .done) return error.PullFailed;
        // What landed has to be the same file: the id this machine computes for
        // it, off its own disk, must be the id it was asked for.
        const landed = try std.fmt.allocPrint(arena, "{s}/{s}", .{ dest, e.stem() });
        var found: ?[]const u8 = null;
        var d = try Io.Dir.cwd().openDir(io, dest, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |ent| {
            if (ent.kind != .file or !std.mem.startsWith(u8, ent.name, e.stem())) continue;
            found = try std.fmt.allocPrint(arena, "{s}/{s}", .{ dest, ent.name });
        }
        const got = found orelse {
            try out.print("{{\"probe\":\"nothing at {s}\"}}\n", .{landed});
            try out.flush();
            return error.PullMissing;
        };
        const here = try catalog.fileId(gpa, io, got);
        try out.print("{{\"probe\":\"pull checked\",\"file\":\"{s}\",\"same_id\":{}}}\n", .{ got, here == id });
        try out.flush();
        if (here != id) return error.PullWrongFile;
    }

    // The image must wait for B's `state` (diff_present) or the router falls
    // back to the chat host; the chat can go at once. A remote B first has to
    // send its catalog, from which the image model is picked.
    h.post(&cfg, .{ .chat_submit = .{ .text = msg } });
    var image_posted = false;
    var picked = !remote_b;
    var catalog_checked = false;
    var image_host: ?*hosts.Slot = null;
    var image_id: wire.ImageId = 0;
    var killed = false;
    var failed = false;
    var turns_seen: u64 = 0;
    var card_seen = false;
    var card_peak_util: f32 = 0;
    var card_peak_mhz: u32 = 0;
    var card_peak_vram: u64 = 0;
    var b_gen_before: wire.HostGen = b.remote.?.gen;
    const deadline = Io.Clock.real.now(io).nanoseconds + @as(i96, @intCast(timeout_s)) * std.time.ns_per_s;

    while (true) {
        if (Io.Clock.real.now(io).nanoseconds > deadline) {
            try out.writeAll("{\"probe\":\"timeout\"}\n");
            try out.flush();
            return error.Timeout;
        }
        h.pump(&cfg);

        if (remote_b and !catalog_checked and b.mirror.scannedOnce()) {
            catalog_checked = true;
            // A remote catalog names files by id and stem, never by path.
            var ids_only = true;
            var ckpts: usize = 0;
            for (b.mirror.catalog.entries) |*e| {
                if (catalog.parseId(e.path) == null or e.name.len == 0 or std.mem.indexOfAny(u8, e.path, "/\\") != null) ids_only = false;
                if (e.ckpt != null) ckpts += 1;
            }
            try out.print("{{\"probe\":\"catalog\",\"host\":\"B\",\"entries\":{d},\"checkpoints\":{d},\"ids_only\":{}}}\n", .{ b.mirror.catalog.entries.len, ckpts, ids_only });
            if (!ids_only) failed = true;
            // The merged view every menu reads. A file only B has must be a row
            // in it, or it is in no menu at all: that is the whole point of
            // merging, and it cannot be seen from one host's catalog.
            {
                var srcs: [models.max_sources]models.Source = undefined;
                var u = try models.build(gpa, h.modelSources(&srcs));
                defer u.deinit();
                var remote_only: usize = 0;
                for (u.rows) |*r| if (r.on != 0 and !models.has(r.on, 0)) {
                    remote_only += 1;
                };
                try out.print("{{\"probe\":\"merged\",\"rows\":{d},\"remote_only\":{d}}}\n", .{ u.cat.entries.len, remote_only });
                // Every file B listed has to be reachable through the merge.
                if (u.cat.entries.len < b.mirror.catalog.entries.len) failed = true;
            }
            // Pick the first checkpoint B offers, as the studio's menu would,
            // and push the settings: B resolves the id to its own file.
            const fams = b.mirror.catalog.families();
            var it = fams.iterator();
            while (it.next()) |fam| if (b.mirror.catalog.firstCheckpoint(fam)) |ci| {
                selection.selectCheckpoint(&cfg, &b.mirror.catalog, b.mirror.catalog.entries[ci].path);
                try out.print("{{\"probe\":\"picked\",\"model\":\"{s}\",\"ref\":\"{s}\"}}\n", .{ b.mirror.catalog.entries[ci].stem(), cfg.diffusion_model.slice() });
                picked = true;
                break;
            };
            if (!picked) {
                try out.writeAll("{\"probe\":\"B offers no checkpoint\"}\n");
                try out.flush();
                return error.NoCheckpoint;
            }
            h.postSettings(&cfg);
        }
        if (!image_posted and picked and b.mirror.state.diff_present) {
            image_posted = true;
            image_host = switch (h.placeRender(&cfg, @intCast(cfg.width), @intCast(cfg.height), @intCast(cfg.steps))) {
                .place => |id| h.slotOf(id),
                .send_model, .nowhere => null,
            };
            if (image_host == null) {
                try out.writeAll("{\"probe\":\"no host will take the render\"}\n");
                try out.flush();
                return error.NoHostForRender;
            }
            image_host.?.post(.{ .img_enqueue = .{
                .client_ref = 1,
                .prompt = prompt,
                .width = @intCast(cfg.width),
                .height = @intCast(cfg.height),
                .steps = @intCast(cfg.steps),
                .seed = 1234,
                .from_studio = true,
            } });
            try out.print("{{\"probe\":\"image posted\",\"host\":\"{s}\"}}\n", .{image_host.?.name});
            if (image_host.? != b) failed = true;
        }
        // What B says about its card: the gate for a host whose GPU has no
        // NVML to ask. The peak matters as much as the first reading, since a
        // meter that only ever reports zero looks the same as a missing one.
        {
            const tel = &b.mirror.telemetry;
            if (tel.gpu_util > card_peak_util) card_peak_util = tel.gpu_util;
            if (tel.gpu_mhz > card_peak_mhz) card_peak_mhz = tel.gpu_mhz;
            if (tel.vram_used > card_peak_vram) card_peak_vram = tel.vram_used;
        }
        if (!card_seen and b.mirror.telemetry.vram_total > 0) {
            card_seen = true;
            const tel = &b.mirror.telemetry;
            try out.print("{{\"probe\":\"card\",\"host\":\"B\",\"have_gpu\":{},\"vram_total_mb\":{d},\"vram_used_mb\":{d},\"vram_proc_mb\":{d},\"gpu_util\":{d:.0},\"gpu_mhz\":{d}}}\n", .{
                tel.have_gpu, tel.vram_total >> 20, tel.vram_used >> 20, tel.vram_proc >> 20, tel.gpu_util, tel.gpu_mhz,
            });
            try out.flush();
        }
        if (a.mirror.turns_ended > turns_seen) {
            turns_seen = a.mirror.turns_ended;
            try out.print("{{\"probe\":\"turn_end\",\"host\":\"a\",\"messages\":{d},\"reply_bytes\":{d}}}\n", .{
                a.mirror.messages.items.len,
                if (a.mirror.messages.items.len > 1) a.mirror.messages.items[1].active().text.items.len else 0,
            });
        }
        // The image, by the ref echoed back, on whichever host has it.
        var im_status: ?wire.ImageStatus = null;
        var im_failure: []const u8 = "";
        var im_step: u32 = 0;
        for (h.slots.items) |s| for (s.mirror.images.items) |*im| if (im.info.client_ref == 1) {
            image_id = im.info.id;
            im_status = im.status();
            im_failure = im.info.failure;
            im_step = im.info.step;
        };
        if (kill_b and !killed and im_status == .generating and im_step >= 2) {
            killed = true;
            try out.print("{{\"probe\":\"killing B\",\"step\":{d}}}\n", .{im_step});
            if (b.remote) |r| if (r.child) |*c| c.kill(io);
        }
        if (killed and b.up() and b.remote.?.gen != b_gen_before) {
            b_gen_before = b.remote.?.gen;
            try out.print("{{\"probe\":\"B respawned\",\"gen\":{d}}}\n", .{b_gen_before});
        }
        try out.flush();

        const image_done = if (im_status) |st| st == .done or st == .failed or st == .canceled else false;
        const b_back = !kill_b or (killed and b.reconnects > 0 and b.up());
        if (turns_seen >= 1 and image_done and b_back) {
            const ok_image = if (kill_b) im_status == .failed and std.mem.eql(u8, im_failure, "HostLost") else im_status == .done;
            const ok_chat = a.mirror.messages.items.len == 2 and a.up() and a.reconnects == 0;
            if (card_seen) try out.print("{{\"probe\":\"card peak\",\"host\":\"B\",\"gpu_util\":{d:.0},\"gpu_mhz\":{d},\"vram_used_mb\":{d}}}\n", .{ card_peak_util, card_peak_mhz, card_peak_vram >> 20 });
            try out.print("{{\"probe\":\"hosts\",\"image\":\"{t}\",\"failure\":\"{s}\",\"chat_messages\":{d},\"a_untouched\":{},\"pass\":{}}}\n", .{
                im_status.?, im_failure, a.mirror.messages.items.len, ok_chat, ok_image and ok_chat and !failed,
            });
            try out.flush();
            if (!(ok_image and ok_chat and !failed)) return error.ProbeFailed;
            return;
        }
        Io.sleep(io, .{ .nanoseconds = 5 * std.time.ns_per_ms }, .real) catch {};
    }
}
