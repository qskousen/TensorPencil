#!/usr/bin/env python3
"""Add a model's embedded chat_template to src/core/assets/jinja/fixtures.json,
with every case's `expected` rendered by REAL jinja2.

The fixture file that `test "jinja: golden fixtures render byte-exact vs jinja2"`
reads is a set of (template, context) -> expected-output triples. This script
appends one template's worth of them, taking the template straight out of a GGUF
so the thing under test is the template a render actually uses, not a
hand-transcribed copy of it. A standalone .jinja file works too, for the
templates we embed as assets rather than read from a checkpoint.

It also RE-RENDERS every pre-existing case and refuses to write if any of them
now disagrees with jinja2. That guards the one thing a generator can silently get
wrong: a jinja2 version whose semantics differ from the one that produced the
existing goldens would quietly move the goalposts for the whole suite.

Needs jinja2 (e.g. /home/qt/genai/ai-toolkit/venv/bin/python).

    python3 tools/gen_jinja_fixtures.py <model.gguf | template.jinja> <template-name>
"""

import json
import struct
import sys
from pathlib import Path

import jinja2

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "src/core/assets/jinja/fixtures.json"


# ── GGUF chat_template extraction ────────────────────────────────────────────
# Only enough of the header parser to reach one string KV. GGUF stores its
# metadata before the tensor table, so this never reads past a few MB.


def read_chat_template(path):
    f = open(path, "rb")

    def raw(n):
        return f.read(n)

    def u32():
        return struct.unpack("<I", raw(4))[0]

    def u64():
        return struct.unpack("<Q", raw(8))[0]

    def string():
        return raw(u64()).decode("utf-8")

    def value(t):
        if t == 0:
            return struct.unpack("<B", raw(1))[0]
        if t == 1:
            return struct.unpack("<b", raw(1))[0]
        if t == 2:
            return struct.unpack("<H", raw(2))[0]
        if t == 3:
            return struct.unpack("<h", raw(2))[0]
        if t == 4:
            return u32()
        if t == 5:
            return struct.unpack("<i", raw(4))[0]
        if t == 6:
            return struct.unpack("<f", raw(4))[0]
        if t == 7:
            return raw(1) != b"\x00"
        if t == 8:
            return string()
        if t == 10:
            return u64()
        if t == 11:
            return struct.unpack("<q", raw(8))[0]
        if t == 12:
            return struct.unpack("<d", raw(8))[0]
        if t == 9:  # array
            et, n = u32(), u64()
            return [value(et) for _ in range(n)]
        raise ValueError(f"unknown gguf value type {t}")

    assert raw(4) == b"GGUF", "not a GGUF file"
    u32()  # version
    u64()  # tensor count
    n_kv = u64()
    for _ in range(n_kv):
        k = string()
        v = value(u32())
        if k == "tokenizer.chat_template":
            return v
    raise SystemExit("no tokenizer.chat_template in that GGUF")


# ── The contexts ─────────────────────────────────────────────────────────────
# Mirrors the existing qwen35 cases (plain / system / multi-turn / thought
# history / images) and then adds the cases that give the Bonsai deltas teeth.
# See the comments on each group: a case that cannot distinguish the template
# variants is decoration, not a test.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

THOUGHT_HISTORY = [
    {"role": "user", "content": "Hi"},
    {
        "role": "assistant",
        "content": "<|channel>thought\nThe user greeted me. I'll greet back.\n<channel|>Hello there!",
    },
    {"role": "user", "content": "Thanks"},
    {
        "role": "assistant",
        "content": "<|channel>thought\nStill polite.\n<channel|>You are welcome.",
    },
    {"role": "user", "content": "Bye"},
]

# The same history in the ChatML dialect, where a prior turn's reasoning is
# stored as `<think>...</think>` INSIDE the assistant content. Qwen templates
# split it back out; the `<|channel>` history above reads to them as opaque
# text, so only this one exercises the extraction and the preserve/drop choice
# that depends on it.
THINK_TAG_HISTORY = [
    {"role": "user", "content": "Hi"},
    {
        "role": "assistant",
        "content": "<think>\nThe user greeted me. I'll greet back.\n</think>\n\nHello there!",
    },
    {"role": "user", "content": "Thanks"},
    {"role": "assistant", "content": "<think>\nStill polite.\n</think>\n\nYou are welcome."},
    {"role": "user", "content": "Bye"},
]

# Tool-call argument values chosen so the two `args_value` spellings DISAGREE.
#   old: `... | tojson if mapping or (sequence and not string) else ... | string`
#   new: `... | string if string else ... | tojson`
# They coincide on strings, dicts, lists and ints; they differ on bool and None,
# where Python's str() gives "True"/"None" and tojson gives "true"/null. Without
# those two the case cannot tell the variants apart.
TOOL_CALL_ARGS = {
    "city": "Paris",
    "opts": {"units": "c"},
    "days": [1, 2],
    "count": 3,
    "verbose": True,
    "note": None,
}


def cases_for(name):
    base = {"bos_token": "", "add_generation_prompt": True}
    out = []

    def add(key, **ctx):
        out.append({"key": f"{name}__{key}", "template": name, "context": {**base, **ctx}})

    user = [{"role": "user", "content": "Hello"}]
    sys_user = [{"role": "system", "content": "Be terse."}] + user
    multi = user + [{"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "More"}]

    # The think_on / think_off / default triples, as for every other template.
    for label, msgs in (("user_only", user), ("sys_user", sys_user), ("multi_turn", multi)):
        add(f"{label}__think_on", messages=msgs, enable_thinking=True)
        add(f"{label}__think_off", messages=msgs, enable_thinking=False)
        add(f"{label}__default", messages=msgs)

    for label in ("think_on", "think_off", "default"):
        kw = {} if label == "default" else {"enable_thinking": label == "think_on"}
        add(f"thought_history__{label}", messages=THOUGHT_HISTORY, **kw)
        add(f"think_tag_history__{label}", messages=THINK_TAG_HISTORY, **kw)

    # Delta 1: `preserve_thinking`. False drops the thoughts of every turn at or
    # before the last user query; true keeps all of them. Which one `absent`
    # means is the template's choice. Needs a MULTI-turn thought history — with
    # one assistant turn the two branches produce the same text — and is run
    # over both dialects, since a template only sees the one it parses.
    for tag, msgs in (("", THOUGHT_HISTORY), ("think_tag_", THINK_TAG_HISTORY)):
        add(f"{tag}preserve_thinking_true", messages=msgs, preserve_thinking=True)
        add(f"{tag}preserve_thinking_false", messages=msgs, preserve_thinking=False)
        add(f"{tag}preserve_thinking_absent", messages=msgs)

    # Delta 2: tool-call argument stringification (see TOOL_CALL_ARGS).
    add("tools_decl", messages=user, tools=TOOLS)
    add(
        "tool_call_args",
        messages=user
        + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": TOOL_CALL_ARGS}}
                ],
            },
            {"role": "tool", "content": "18C"},
        ],
        tools=TOOLS,
    )

    # Vision: the content-list form, which drives the render_content macro.
    add(
        "img_text",
        messages=[{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image"}]}],
        enable_thinking=False,
    )
    add("img_only", messages=[{"role": "user", "content": [{"type": "image"}]}], enable_thinking=False)
    add(
        "img_two",
        messages=[{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare"}]}],
        enable_thinking=False,
    )
    return out


def env():
    # trim_blocks/lstrip_blocks stay OFF: the templates use explicit `{%-`/`-%}`
    # whitespace control, which is what both llama.cpp's minja and our engine do.
    #
    # The default (lenient) Undefined, NOT StrictUndefined: these templates test
    # `{%- if tools %}` on a name the caller may never bind, so strict mode raises
    # where a real chat frontend renders. It is also what produced the existing
    # goldens — the guard below would reject this script otherwise.
    # Every setting here is pinned by the "reproduces all existing goldens" guard
    # below rather than chosen: `keep_trailing_newline` defaults to False, and the
    # llama goldens have no trailing newline, so True fails 15 of them.
    e = jinja2.Environment()
    e.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    return e


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_path, name = Path(sys.argv[1]), sys.argv[2]

    if src_path.suffix == ".jinja":
        tmpl_src = src_path.read_text()
    else:
        tmpl_src = read_chat_template(src_path)
    data = json.loads(FIXTURES.read_text())
    e = env()

    # Guard: this jinja2 must reproduce every existing golden before it is
    # allowed to author new ones.
    drift = []
    for c in data["cases"]:
        got = e.from_string(data["templates"][c["template"]]).render(**c["context"])
        if got != c["expected"]:
            drift.append(c["key"])
    if drift:
        raise SystemExit(
            f"this jinja2 ({jinja2.__version__}) disagrees with {len(drift)} existing "
            f"golden(s), e.g. {drift[:3]}; refusing to write"
        )
    print(f"jinja2 {jinja2.__version__} reproduces all {len(data['cases'])} existing goldens")

    data["templates"][name] = tmpl_src
    data["cases"] = [c for c in data["cases"] if c["template"] != name]
    t = e.from_string(tmpl_src)
    added = 0
    for c in cases_for(name):
        c["expected"] = t.render(**c["context"])
        data["cases"].append(c)
        added += 1

    # Teeth check for the two deltas. Which WAY a template resolves them is the
    # template's business (froggeric's fixed Qwen template preserves thoughts by
    # default where the official Qwen 3.6 one drops them); what the corpus must
    # do is tell the settings apart at all.
    by_key = {c["key"]: c["expected"] for c in data["cases"]}
    distinguished = False
    for tag in ("", "think_tag_"):
        on, off, absent = (by_key[f"{name}__{tag}preserve_thinking_{k}"] for k in ("true", "false", "absent"))
        if on == off:
            continue
        distinguished = True
        if absent == on:
            print(f"  {tag or 'channel_'}history: preserve_thinking defaults to true (thoughts kept)")
        elif absent == off:
            print(f"  {tag or 'channel_'}history: preserve_thinking defaults to false (thoughts dropped)")
        else:
            raise SystemExit(f"{name}__{tag}preserve_thinking_absent matches neither true nor false")
    if not distinguished:
        raise SystemExit(f"no teeth: no {name} thought history tells preserve_thinking true from false")

    # Arguments render as `<parameter=k>\n<value>\n</parameter>`, so the
    # discriminator is the bare value: `tojson` gives true/null where Python's
    # str() (the older spelling) gives True/None. Both spellings are in the
    # wild; the case has teeth as long as the render commits to one of them.
    args_out = by_key[f"{name}__tool_call_args"]
    for key, json_val, str_val in (("verbose", "true", "True"), ("note", "null", "None")):
        got = [v for v in (json_val, str_val) if f"<parameter={key}>\n{v}\n" in args_out]
        if len(got) != 1:
            raise SystemExit(
                f"no teeth: <parameter={key}> is neither {json_val!r} nor {str_val!r} in the "
                f"tool-call render — the args spelling is not pinned"
            )
        print(f"  tool-call {key} renders as {got[0]!r}")

    FIXTURES.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {added} '{name}' cases to {FIXTURES.relative_to(REPO)} ({len(data['cases'])} total)")


if __name__ == "__main__":
    main()
