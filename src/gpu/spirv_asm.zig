//! A tiny SPIR-V text assembler, written in Zig.
//!
//! Kernels (see `coopmat.zig`) are authored as readable SPIR-V *assembly text*
//! — the same `%id = OpName operands` syntax `spirv-as`/`spirv-dis` use — and
//! this module turns that text into the binary word stream a Vulkan driver
//! consumes. It replaces the previous approach of emitting words by hand with
//! manual id pre-allocation and magic opcode numbers.
//!
//! Scope: this is NOT a general SPIR-V assembler. It covers the opcodes and
//! named enum tokens this codebase actually uses; the enum table is a flat
//! name->value map (see `enum_tokens`), which works only because our usage of
//! each name is unambiguous. Unknown mnemonics/tokens are a hard error, so
//! gaps surface immediately rather than miscompiling. Correctness is
//! cross-checked against the real `spirv-as` (byte-identical output, since both
//! number ids by first occurrence) and against `spirv-val`.
//!
//! Grammar accepted, one instruction per line:
//!   %result = OpName operand...        (ops with a result id)
//!             OpName operand...        (ops without a result id)
//! Operands:
//!   %name          - an id reference (interned; allocated on first mention,
//!                    so forward references to labels/globals just work)
//!   "text"         - a literal string (null-terminated, word-padded)
//!   123 / 0x7f     - a literal integer word (decimal or hex, optional '-')
//!   Name           - a named enum token, resolved via `enum_tokens`
//! `;` starts a comment to end of line. Blank lines are ignored.
//!
//! For an op that carries a result *type* (e.g. `OpLoad`, `OpIAdd`), write it
//! as `%r = OpLoad %type %ptr`: the assembler knows from the opcode table to
//! emit <result-type> then <result-id> then the rest, matching SPIR-V binary
//! layout. For type/label ops (`OpTypeInt`, `OpLabel`) the result id has no
//! preceding type and is emitted first.

const std = @import("std");

pub const magic: u32 = 0x0723_0203;

/// SPIR-V version words for the versions we target.
pub const version_1_3: u32 = 0x0001_0300;
pub const version_1_4: u32 = 0x0001_0400;
pub const version_1_5: u32 = 0x0001_0500;
pub const version_1_6: u32 = 0x0001_0600;

pub const Error = error{
    UnknownOpcode,
    UnknownToken,
    BadInteger,
    UnterminatedString,
    MissingResult,
    MissingResultType,
    OutOfMemory,
};

/// Where the result id sits (if any) relative to the operands.
const ResultKind = enum {
    none, // no result id (OpStore, OpBranch, OpDecorate, ...)
    id, // result id, no result type (OpTypeInt, OpLabel, OpExtInstImport, ...)
    typed, // result type then result id (OpLoad, OpIAdd, OpVariable, ...)
};

const OpInfo = struct { opcode: u16, result: ResultKind };

/// Opcode table: mnemonic -> {opcode number, result kind}. Grow as kernels
/// use new instructions; a missing entry is `error.UnknownOpcode`.
const opcodes = std.StaticStringMap(OpInfo).initComptime(.{
    // Debug / module structure (no result)
    .{ "OpCapability", OpInfo{ .opcode = 17, .result = .none } },
    .{ "OpExtension", OpInfo{ .opcode = 10, .result = .none } },
    .{ "OpMemoryModel", OpInfo{ .opcode = 14, .result = .none } },
    .{ "OpEntryPoint", OpInfo{ .opcode = 15, .result = .none } },
    .{ "OpExecutionMode", OpInfo{ .opcode = 16, .result = .none } },
    .{ "OpName", OpInfo{ .opcode = 5, .result = .none } },
    .{ "OpMemberName", OpInfo{ .opcode = 6, .result = .none } },
    .{ "OpDecorate", OpInfo{ .opcode = 71, .result = .none } },
    .{ "OpMemberDecorate", OpInfo{ .opcode = 72, .result = .none } },
    // Ext inst
    .{ "OpExtInstImport", OpInfo{ .opcode = 11, .result = .id } },
    .{ "OpExtInst", OpInfo{ .opcode = 12, .result = .typed } },
    // Types (result id, no type)
    .{ "OpTypeVoid", OpInfo{ .opcode = 19, .result = .id } },
    .{ "OpTypeBool", OpInfo{ .opcode = 20, .result = .id } },
    .{ "OpTypeInt", OpInfo{ .opcode = 21, .result = .id } },
    .{ "OpTypeFloat", OpInfo{ .opcode = 22, .result = .id } },
    .{ "OpTypeVector", OpInfo{ .opcode = 23, .result = .id } },
    .{ "OpTypeMatrix", OpInfo{ .opcode = 24, .result = .id } },
    .{ "OpTypeArray", OpInfo{ .opcode = 28, .result = .id } },
    .{ "OpTypeRuntimeArray", OpInfo{ .opcode = 29, .result = .id } },
    .{ "OpTypeStruct", OpInfo{ .opcode = 30, .result = .id } },
    .{ "OpTypePointer", OpInfo{ .opcode = 32, .result = .id } },
    .{ "OpTypeFunction", OpInfo{ .opcode = 33, .result = .id } },
    .{ "OpTypeCooperativeMatrixKHR", OpInfo{ .opcode = 4456, .result = .id } },
    // Constants (result + type)
    .{ "OpConstantTrue", OpInfo{ .opcode = 41, .result = .typed } },
    .{ "OpConstantFalse", OpInfo{ .opcode = 42, .result = .typed } },
    .{ "OpConstant", OpInfo{ .opcode = 43, .result = .typed } },
    .{ "OpConstantComposite", OpInfo{ .opcode = 44, .result = .typed } },
    .{ "OpConstantNull", OpInfo{ .opcode = 46, .result = .typed } },
    // Memory / function (result + type unless noted)
    .{ "OpFunction", OpInfo{ .opcode = 54, .result = .typed } },
    .{ "OpFunctionParameter", OpInfo{ .opcode = 55, .result = .typed } },
    .{ "OpFunctionEnd", OpInfo{ .opcode = 56, .result = .none } },
    .{ "OpFunctionCall", OpInfo{ .opcode = 57, .result = .typed } },
    .{ "OpVariable", OpInfo{ .opcode = 59, .result = .typed } },
    .{ "OpLoad", OpInfo{ .opcode = 61, .result = .typed } },
    .{ "OpStore", OpInfo{ .opcode = 62, .result = .none } },
    .{ "OpAccessChain", OpInfo{ .opcode = 65, .result = .typed } },
    // Composite
    .{ "OpVectorShuffle", OpInfo{ .opcode = 79, .result = .typed } },
    .{ "OpCompositeConstruct", OpInfo{ .opcode = 80, .result = .typed } },
    .{ "OpCompositeExtract", OpInfo{ .opcode = 81, .result = .typed } },
    // Conversion
    .{ "OpConvertFToU", OpInfo{ .opcode = 109, .result = .typed } },
    .{ "OpConvertFToS", OpInfo{ .opcode = 110, .result = .typed } },
    .{ "OpConvertSToF", OpInfo{ .opcode = 111, .result = .typed } },
    .{ "OpConvertUToF", OpInfo{ .opcode = 112, .result = .typed } },
    .{ "OpUConvert", OpInfo{ .opcode = 113, .result = .typed } },
    .{ "OpSConvert", OpInfo{ .opcode = 114, .result = .typed } },
    .{ "OpFConvert", OpInfo{ .opcode = 115, .result = .typed } },
    .{ "OpBitcast", OpInfo{ .opcode = 124, .result = .typed } },
    // Arithmetic
    .{ "OpSNegate", OpInfo{ .opcode = 126, .result = .typed } },
    .{ "OpFNegate", OpInfo{ .opcode = 127, .result = .typed } },
    .{ "OpIAdd", OpInfo{ .opcode = 128, .result = .typed } },
    .{ "OpFAdd", OpInfo{ .opcode = 129, .result = .typed } },
    .{ "OpISub", OpInfo{ .opcode = 130, .result = .typed } },
    .{ "OpFSub", OpInfo{ .opcode = 131, .result = .typed } },
    .{ "OpIMul", OpInfo{ .opcode = 132, .result = .typed } },
    .{ "OpFMul", OpInfo{ .opcode = 133, .result = .typed } },
    .{ "OpUDiv", OpInfo{ .opcode = 134, .result = .typed } },
    .{ "OpSDiv", OpInfo{ .opcode = 135, .result = .typed } },
    .{ "OpFDiv", OpInfo{ .opcode = 136, .result = .typed } },
    .{ "OpUMod", OpInfo{ .opcode = 137, .result = .typed } },
    .{ "OpSRem", OpInfo{ .opcode = 138, .result = .typed } },
    .{ "OpSMod", OpInfo{ .opcode = 139, .result = .typed } },
    .{ "OpFRem", OpInfo{ .opcode = 140, .result = .typed } },
    .{ "OpFMod", OpInfo{ .opcode = 141, .result = .typed } },
    // Bitwise / shift
    .{ "OpShiftRightLogical", OpInfo{ .opcode = 194, .result = .typed } },
    .{ "OpShiftRightArithmetic", OpInfo{ .opcode = 195, .result = .typed } },
    .{ "OpShiftLeftLogical", OpInfo{ .opcode = 196, .result = .typed } },
    .{ "OpBitwiseOr", OpInfo{ .opcode = 197, .result = .typed } },
    .{ "OpBitwiseXor", OpInfo{ .opcode = 198, .result = .typed } },
    .{ "OpBitwiseAnd", OpInfo{ .opcode = 199, .result = .typed } },
    .{ "OpNot", OpInfo{ .opcode = 200, .result = .typed } },
    // Relational / logical
    .{ "OpLogicalOr", OpInfo{ .opcode = 166, .result = .typed } },
    .{ "OpLogicalAnd", OpInfo{ .opcode = 167, .result = .typed } },
    .{ "OpLogicalNot", OpInfo{ .opcode = 168, .result = .typed } },
    .{ "OpSelect", OpInfo{ .opcode = 169, .result = .typed } },
    .{ "OpIEqual", OpInfo{ .opcode = 170, .result = .typed } },
    .{ "OpINotEqual", OpInfo{ .opcode = 171, .result = .typed } },
    .{ "OpUGreaterThan", OpInfo{ .opcode = 172, .result = .typed } },
    .{ "OpSGreaterThan", OpInfo{ .opcode = 173, .result = .typed } },
    .{ "OpUGreaterThanEqual", OpInfo{ .opcode = 174, .result = .typed } },
    .{ "OpSGreaterThanEqual", OpInfo{ .opcode = 175, .result = .typed } },
    .{ "OpULessThan", OpInfo{ .opcode = 176, .result = .typed } },
    .{ "OpSLessThan", OpInfo{ .opcode = 177, .result = .typed } },
    .{ "OpULessThanEqual", OpInfo{ .opcode = 178, .result = .typed } },
    .{ "OpSLessThanEqual", OpInfo{ .opcode = 179, .result = .typed } },
    .{ "OpFOrdEqual", OpInfo{ .opcode = 180, .result = .typed } },
    .{ "OpFOrdLessThan", OpInfo{ .opcode = 184, .result = .typed } },
    .{ "OpFOrdGreaterThan", OpInfo{ .opcode = 186, .result = .typed } },
    .{ "OpFOrdLessThanEqual", OpInfo{ .opcode = 188, .result = .typed } },
    .{ "OpFOrdGreaterThanEqual", OpInfo{ .opcode = 190, .result = .typed } },
    // Control flow
    .{ "OpPhi", OpInfo{ .opcode = 245, .result = .typed } },
    .{ "OpLoopMerge", OpInfo{ .opcode = 246, .result = .none } },
    .{ "OpSelectionMerge", OpInfo{ .opcode = 247, .result = .none } },
    .{ "OpLabel", OpInfo{ .opcode = 248, .result = .id } },
    .{ "OpBranch", OpInfo{ .opcode = 249, .result = .none } },
    .{ "OpBranchConditional", OpInfo{ .opcode = 250, .result = .none } },
    .{ "OpSwitch", OpInfo{ .opcode = 251, .result = .none } },
    .{ "OpReturn", OpInfo{ .opcode = 253, .result = .none } },
    .{ "OpReturnValue", OpInfo{ .opcode = 254, .result = .none } },
    .{ "OpUnreachable", OpInfo{ .opcode = 255, .result = .none } },
    // Barriers
    .{ "OpControlBarrier", OpInfo{ .opcode = 224, .result = .none } },
    .{ "OpMemoryBarrier", OpInfo{ .opcode = 225, .result = .none } },
    // Cooperative matrix KHR
    .{ "OpCooperativeMatrixLoadKHR", OpInfo{ .opcode = 4457, .result = .typed } },
    .{ "OpCooperativeMatrixStoreKHR", OpInfo{ .opcode = 4458, .result = .none } },
    .{ "OpCooperativeMatrixMulAddKHR", OpInfo{ .opcode = 4459, .result = .typed } },
    .{ "OpCooperativeMatrixLengthKHR", OpInfo{ .opcode = 4460, .result = .typed } },
});

/// Named enum tokens -> value. Flat and codebase-scoped: only names whose use
/// here is unambiguous. Anything not listed can always be written as its
/// numeric literal instead.
const enum_tokens = std.StaticStringMap(u32).initComptime(.{
    // Capabilities
    .{ "Matrix", 0 },
    .{ "Shader", 1 },
    .{ "Float16", 9 },
    .{ "Float64", 10 },
    .{ "Int64", 11 },
    .{ "Int16", 22 },
    .{ "Int8", 39 },
    .{ "GroupNonUniform", 61 },
    .{ "StorageBuffer16BitAccess", 4433 },
    .{ "StorageBuffer8BitAccess", 4448 },
    .{ "VulkanMemoryModel", 5345 },
    .{ "VulkanMemoryModelDeviceScope", 5346 },
    .{ "CooperativeMatrixKHR", 6022 },
    // Addressing model
    .{ "Logical", 0 },
    // Memory model
    .{ "GLSL450", 1 },
    .{ "Vulkan", 3 },
    // Execution model
    .{ "GLCompute", 5 },
    .{ "Kernel", 6 },
    // Execution mode
    .{ "LocalSize", 17 },
    // Function control
    .{ "None", 0 },
    .{ "Inline", 1 },
    .{ "DontInline", 2 },
    // Storage classes
    .{ "UniformConstant", 0 },
    .{ "Input", 1 },
    .{ "Uniform", 2 },
    .{ "Output", 3 },
    .{ "Workgroup", 4 },
    .{ "Private", 6 },
    .{ "Function", 7 },
    .{ "PushConstant", 9 },
    .{ "StorageBuffer", 12 },
    // Decorations
    .{ "Block", 2 },
    .{ "ArrayStride", 6 },
    .{ "BuiltIn", 11 },
    .{ "Binding", 33 },
    .{ "DescriptorSet", 34 },
    .{ "Offset", 35 },
    // BuiltIns
    .{ "NumWorkgroups", 24 },
    .{ "WorkgroupId", 26 },
    .{ "LocalInvocationId", 27 },
    .{ "GlobalInvocationId", 28 },
    .{ "LocalInvocationIndex", 29 },
    .{ "SubgroupSize", 36 },
    .{ "NumSubgroups", 38 },
    .{ "SubgroupId", 40 },
    .{ "SubgroupLocalInvocationId", 41 },
});

/// The assembler. Accumulates instruction words across one or more `add`
/// calls, then `finish` prepends the 5-word header (with the computed id
/// bound) and returns the module bytes.
pub const Assembler = struct {
    gpa: std.mem.Allocator,
    version: u32,
    body: std.ArrayList(u32) = .empty,
    names: std.StringHashMapUnmanaged(u32) = .empty,
    next_id: u32 = 1,

    pub fn init(gpa: std.mem.Allocator, version: u32) Assembler {
        return .{ .gpa = gpa, .version = version };
    }

    pub fn deinit(self: *Assembler) void {
        var it = self.names.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.names.deinit(self.gpa);
        self.body.deinit(self.gpa);
    }

    /// Resolve `%name` (leading '%' already stripped) to its id, allocating on
    /// first mention so forward references resolve naturally.
    fn idOf(self: *Assembler, name: []const u8) Error!u32 {
        const gop = try self.names.getOrPut(self.gpa, name);
        if (!gop.found_existing) {
            gop.key_ptr.* = try self.gpa.dupe(u8, name);
            gop.value_ptr.* = self.next_id;
            self.next_id += 1;
        }
        return gop.value_ptr.*;
    }

    /// Assemble a block of SPIR-V assembly text, appending its instructions.
    pub fn add(self: *Assembler, text: []const u8) Error!void {
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |raw| {
            const line = stripComment(raw);
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;
            try self.assembleLine(trimmed);
        }
    }

    fn assembleLine(self: *Assembler, line: []const u8) Error!void {
        var operands: std.ArrayList(u32) = .empty;
        defer operands.deinit(self.gpa);

        var toks = Tokenizer{ .s = line };

        // Optional "%result =" prefix.
        var result_name: ?[]const u8 = null;
        const first = (try toks.next(self.gpa, &operands)) orelse return;
        if (first == .id_name) {
            // Peek for '='.
            const save = toks;
            const maybe_eq = try toks.next(self.gpa, &operands);
            if (maybe_eq != null and maybe_eq.? == .equals) {
                result_name = first.id_name;
            } else {
                // Not an assignment: the line starts with an id operand (rare;
                // no such instruction here). Restore and treat `first` as the
                // mnemonic slot — but a leading id without '=' is malformed.
                toks = save;
                return Error.UnknownOpcode;
            }
        }

        // Mnemonic.
        const mnem_tok = if (result_name == null) first else (try toks.next(self.gpa, &operands)) orelse return Error.UnknownOpcode;
        if (mnem_tok != .word) return Error.UnknownOpcode;
        const info = opcodes.get(mnem_tok.word) orelse return Error.UnknownOpcode;

        if (result_name == null and info.result != .none) return Error.MissingResult;
        if (result_name != null and info.result == .none) {
            // A result was given for a no-result op — malformed.
            return Error.UnknownOpcode;
        }

        // Collect operand words. For `.typed`, the first parsed operand is the
        // result type; we insert result id after it. For `.id`, result id goes
        // first. For `.none`, no result id.
        var words: std.ArrayList(u32) = .empty;
        defer words.deinit(self.gpa);

        const result_id: u32 = if (result_name) |rn| try self.idOf(rn) else 0;

        // Parse remaining operand tokens into a flat word list.
        var parsed: std.ArrayList(u32) = .empty;
        defer parsed.deinit(self.gpa);
        while (try toks.next(self.gpa, null)) |t| {
            switch (t) {
                .word => |w| try parsed.append(self.gpa, try self.resolveWord(w)),
                .id_name => |n| try parsed.append(self.gpa, try self.idOf(n)),
                .int => |v| try parsed.append(self.gpa, v),
                .string => |bytes| {
                    try appendString(self.gpa, &parsed, bytes);
                    self.gpa.free(bytes);
                },
                .equals => return Error.UnknownOpcode,
            }
        }

        switch (info.result) {
            .none => try words.appendSlice(self.gpa, parsed.items),
            .id => {
                try words.append(self.gpa, result_id);
                try words.appendSlice(self.gpa, parsed.items);
            },
            .typed => {
                if (parsed.items.len < 1) return Error.MissingResultType;
                try words.append(self.gpa, parsed.items[0]); // result type
                try words.append(self.gpa, result_id);
                try words.appendSlice(self.gpa, parsed.items[1..]);
            },
        }

        const word_count: u32 = @intCast(words.items.len + 1);
        try self.body.append(self.gpa, (word_count << 16) | info.opcode);
        try self.body.appendSlice(self.gpa, words.items);
    }

    /// A bare word operand is either a named enum token or (defensively) fails.
    fn resolveWord(self: *Assembler, w: []const u8) Error!u32 {
        _ = self;
        return enum_tokens.get(w) orelse Error.UnknownToken;
    }

    /// Finalize: emit header + body. Caller owns the returned bytes.
    pub fn finish(self: *Assembler) Error![]align(4) u8 {
        const total = 5 + self.body.items.len;
        const out = try self.gpa.alignedAlloc(u8, .of(u32), total * 4);
        const w: []align(4) u32 = @alignCast(std.mem.bytesAsSlice(u32, out));
        w[0] = magic;
        w[1] = self.version;
        w[2] = 0; // generator magic number
        w[3] = self.next_id; // id bound (max id + 1)
        w[4] = 0; // schema
        @memcpy(w[5..], self.body.items);
        return out;
    }
};

/// One-shot convenience: assemble a single text block into a module.
pub fn assemble(gpa: std.mem.Allocator, version: u32, text: []const u8) Error![]align(4) u8 {
    var a = Assembler.init(gpa, version);
    defer a.deinit();
    try a.add(text);
    return a.finish();
}

/// Like `assemble`, but every non-allocation error (unknown opcode/token,
/// malformed operand) is a bug in the kernel text — panic rather than
/// propagate. Lets callers keep a clean `error{OutOfMemory}` set.
pub fn assembleChecked(gpa: std.mem.Allocator, version: u32, text: []const u8) error{OutOfMemory}![]align(4) u8 {
    return assemble(gpa, version, text) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => std.debug.panic("spirv_asm: malformed kernel text: {s}", .{@errorName(e)}),
    };
}

fn stripComment(line: []const u8) []const u8 {
    // ';' starts a comment, but not inside a string literal.
    var in_str = false;
    for (line, 0..) |c, i| {
        if (c == '"') in_str = !in_str;
        if (c == ';' and !in_str) return line[0..i];
    }
    return line;
}

/// Encode a SPIR-V literal string: bytes, null terminator, zero-padded up to a
/// word boundary, little-endian.
fn appendString(gpa: std.mem.Allocator, out: *std.ArrayList(u32), s: []const u8) Error!void {
    const n_words = s.len / 4 + 1; // +1 guarantees at least one null byte
    var i: usize = 0;
    while (i < n_words * 4) : (i += 4) {
        var word: u32 = 0;
        inline for (0..4) |b| {
            if (i + b < s.len) word |= @as(u32, s[i + b]) << (8 * @as(u5, @intCast(b)));
        }
        try out.append(gpa, word);
    }
}

const Token = union(enum) {
    word: []const u8, // bare mnemonic or enum token
    id_name: []const u8, // %name (without '%')
    int: u32, // integer literal word
    string: []u8, // owned decoded string bytes
    equals: void,
};

/// Whitespace-splitting tokenizer that keeps quoted strings intact.
const Tokenizer = struct {
    s: []const u8,
    i: usize = 0,

    /// `sink` is unused scratch kept for API symmetry with callers that want to
    /// free owned tokens; pass null. Strings are heap-decoded and owned by the
    /// caller (freed after use above).
    fn next(self: *Tokenizer, gpa: std.mem.Allocator, sink: ?*std.ArrayList(u32)) Error!?Token {
        _ = sink;
        // Skip whitespace.
        while (self.i < self.s.len and (self.s[self.i] == ' ' or self.s[self.i] == '\t')) self.i += 1;
        if (self.i >= self.s.len) return null;

        const c = self.s[self.i];
        if (c == '=') {
            self.i += 1;
            return Token{ .equals = {} };
        }
        if (c == '"') {
            self.i += 1;
            const start = self.i;
            while (self.i < self.s.len and self.s[self.i] != '"') self.i += 1;
            if (self.i >= self.s.len) return Error.UnterminatedString;
            const bytes = self.s[start..self.i];
            self.i += 1; // closing quote
            return Token{ .string = try gpa.dupe(u8, bytes) };
        }

        const start = self.i;
        while (self.i < self.s.len and self.s[self.i] != ' ' and self.s[self.i] != '\t' and self.s[self.i] != '=') self.i += 1;
        const tok = self.s[start..self.i];

        if (tok[0] == '%') return Token{ .id_name = tok[1..] };
        if (isIntLiteral(tok)) return Token{ .int = try parseInt(tok) };
        return Token{ .word = tok };
    }
};

fn isIntLiteral(tok: []const u8) bool {
    if (tok.len == 0) return false;
    var s = tok;
    if (s[0] == '-' or s[0] == '+') s = s[1..];
    if (s.len == 0) return false;
    if (std.mem.startsWith(u8, s, "0x") or std.mem.startsWith(u8, s, "0X")) return s.len > 2;
    for (s) |ch| if (ch < '0' or ch > '9') return false;
    return true;
}

fn parseInt(tok: []const u8) Error!u32 {
    // Accept decimal (possibly negative -> two's complement) and hex.
    if (std.mem.startsWith(u8, tok, "0x") or std.mem.startsWith(u8, tok, "0X")) {
        return std.fmt.parseInt(u32, tok[2..], 16) catch return Error.BadInteger;
    }
    if (tok[0] == '-') {
        const v = std.fmt.parseInt(i64, tok, 10) catch return Error.BadInteger;
        return @bitCast(@as(i32, @intCast(v)));
    }
    return std.fmt.parseInt(u32, tok, 10) catch return Error.BadInteger;
}

test "assembles a minimal module with header and bound" {
    const gpa = std.testing.allocator;
    const text =
        \\OpCapability Shader
        \\%void = OpTypeVoid
        \\%u32 = OpTypeInt 32 0
        \\%c5 = OpConstant %u32 5
    ;
    const spv = try assemble(gpa, version_1_5, text);
    defer gpa.free(spv);
    const w = std.mem.bytesAsSlice(u32, spv);
    try std.testing.expectEqual(magic, w[0]);
    try std.testing.expectEqual(version_1_5, w[1]);
    // ids: void=1, u32=2, c5=3 -> bound = 4
    try std.testing.expectEqual(@as(u32, 4), w[3]);
    // OpCapability Shader: wordcount 2, opcode 17
    try std.testing.expectEqual((@as(u32, 2) << 16) | 17, w[5]);
    try std.testing.expectEqual(@as(u32, 1), w[6]); // Shader
    // OpTypeVoid: wc 2, opcode 19, result id 1
    try std.testing.expectEqual((@as(u32, 2) << 16) | 19, w[7]);
    try std.testing.expectEqual(@as(u32, 1), w[8]);
    // OpTypeInt: wc 4, opcode 21, result 2, width 32, signed 0
    try std.testing.expectEqual((@as(u32, 4) << 16) | 21, w[9]);
    try std.testing.expectEqual(@as(u32, 2), w[10]);
    try std.testing.expectEqual(@as(u32, 32), w[11]);
    try std.testing.expectEqual(@as(u32, 0), w[12]);
    // OpConstant: typed -> type first (u32=2), then result (c5=3), then value 5
    try std.testing.expectEqual((@as(u32, 4) << 16) | 43, w[13]);
    try std.testing.expectEqual(@as(u32, 2), w[14]);
    try std.testing.expectEqual(@as(u32, 3), w[15]);
    try std.testing.expectEqual(@as(u32, 5), w[16]);
}

test "string literals encode null-terminated and word-padded" {
    const gpa = std.testing.allocator;
    // OpEntryPoint GLCompute %main "main"  (interface omitted for the test)
    const spv = try assemble(gpa, version_1_5,
        \\OpEntryPoint GLCompute %main "main"
    );
    defer gpa.free(spv);
    const w = std.mem.bytesAsSlice(u32, spv);
    // header(5) + instruction. opcode 15. operands: GLCompute(5), %main id, "main"(2 words)
    try std.testing.expectEqual((@as(u32, 5) << 16) | 15, w[5]);
    try std.testing.expectEqual(@as(u32, 5), w[6]); // GLCompute
    try std.testing.expectEqual(@as(u32, 1), w[7]); // %main = id 1
    try std.testing.expectEqual(std.mem.bytesToValue(u32, "main"), w[8]);
    try std.testing.expectEqual(@as(u32, 0), w[9]); // null pad word
}

test "hex, negative, and forward-referenced ids" {
    const gpa = std.testing.allocator;
    const spv = try assemble(gpa, version_1_5,
        \\OpBranch %later
        \\%later = OpLabel
        \\%v = OpTypeInt 32 1
        \\%c = OpConstant %v 0xff
        \\%n = OpConstant %v -1
    );
    defer gpa.free(spv);
    const w = std.mem.bytesAsSlice(u32, spv);
    // OpBranch %later: %later is first mention -> id 1
    try std.testing.expectEqual((@as(u32, 2) << 16) | 249, w[5]);
    try std.testing.expectEqual(@as(u32, 1), w[6]);
    // OpLabel result -> id 1 (same forward ref resolved)
    try std.testing.expectEqual((@as(u32, 2) << 16) | 248, w[7]);
    try std.testing.expectEqual(@as(u32, 1), w[8]);
    // The last two instructions are OpConstant %v 0xff and OpConstant %v -1,
    // 4 words each: [header, type, result, value]. So the -1 value is the very
    // last word, and 0xff is the last word of the preceding constant (len-5).
    try std.testing.expectEqual(@as(u32, 0xff), w[w.len - 5]);
    try std.testing.expectEqual(@as(u32, 0xffff_ffff), w[w.len - 1]);
}
