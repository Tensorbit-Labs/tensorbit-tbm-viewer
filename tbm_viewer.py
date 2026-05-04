#!/usr/bin/env python3
"""
Tensorbit Model (.tbm) Container Viewer

A lightweight GUI for inspecting .tbm model container files.
Parses the JSON index at the file tail and provides per-tensor
metadata inspection with data previews (first N values only to
avoid loading gigabytes into memory).
"""

import struct
import json
import os
import re
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional


# ──────────────────────────────────────────────
# .tbm format constants
# ──────────────────────────────────────────────
TB_HEADER_SIZE = 4096
TB_MAGIC = 0x31304254

DTYPE_SIZES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp64": 8}


def parse_tbm(path: str) -> Optional[dict]:
    """Parse a .tbm file and return its JSON index dictionary.

    The .tbm format is:
        [concatenated .tb blobs] [UTF-8 JSON index] [4-byte LE uint32: json_len]

    Returns None on parse failure.
    """
    try:
        file_size = os.path.getsize(path)
        if file_size < 4:
            return None

        with open(path, "rb") as f:
            # --- read 4-byte tail: index length ---
            f.seek(-4, os.SEEK_END)
            (idx_len,) = struct.unpack("<I", f.read(4))
            if idx_len == 0 or idx_len > file_size - 4:
                return None

            # --- read JSON index ---
            json_start = file_size - 4 - idx_len
            f.seek(json_start)
            raw = f.read(idx_len)
            index = json.loads(raw.decode("utf-8"))
            return index

    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return None


def preview_tensor(path: str, offset: int, num_weights: int,
                   dtype: str, count: int = 128) -> Optional[list[float]]:
    """Read the first `count` weight values from a tensor within a .tbm file.

    Returns a list of floats or None on failure.
    """
    elem_size = DTYPE_SIZES.get(dtype, 4)
    data_offset = offset + TB_HEADER_SIZE
    read_count = min(count, num_weights)
    read_bytes = read_count * elem_size

    try:
        with open(path, "rb") as f:
            f.seek(data_offset)
            blob = f.read(read_bytes)
            if len(blob) < read_bytes:
                return None

        if dtype == "fp32":
            return list(struct.unpack(f"<{read_count}f", blob))
        elif dtype == "fp16":
            vals = []
            for i in range(read_count):
                u16 = struct.unpack_from("<H", blob, i * 2)[0]
                sign = (u16 >> 15) & 1
                exp = (u16 >> 10) & 0x1F
                mant = u16 & 0x3FF
                if exp == 0:
                    val = ((-1) ** sign) * (2 ** (-14)) * (mant / 1024.0)
                elif exp == 31:
                    val = float("nan") if mant != 0 else float("inf") * ((-1) ** sign)
                else:
                    val = ((-1) ** sign) * (2 ** (exp - 15)) * (1 + mant / 1024.0)
                vals.append(val)
            return vals
        elif dtype == "bf16":
            vals = []
            for i in range(read_count):
                u16 = struct.unpack_from("<H", blob, i * 2)[0]
                u32 = u16 << 16
                (f,) = struct.unpack("<f", struct.pack("<I", u32))
                vals.append(f)
            return vals
        elif dtype == "fp64":
            return list(struct.unpack(f"<{read_count}d", blob))
        else:
            return None

    except (OSError, struct.error):
        return None


def preview_masks(path: str, offset: int, num_mask_bytes: int,
                  count: int = 256) -> Optional[list[int]]:
    """Read the first `count` mask bytes from a tensor within a .tbm file.

    Returns a list of integers (0-255) or None on failure.
    """
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            hdr = f.read(TB_HEADER_SIZE)
            if len(hdr) < TB_HEADER_SIZE:
                return None

            # TBHeader.masks_offset is at byte offset 40
            masks_offset = struct.unpack_from("<Q", hdr, 40)[0]
            mask_start = offset + masks_offset

            read_count = min(count, num_mask_bytes)
            f.seek(mask_start)
            blob = f.read(read_count)
            return list(blob)

    except (OSError, struct.error):
        return None


# ──────────────────────────────────────────────
# GUI Application
# ──────────────────────────────────────────────


class TbmViewerApp:
    """Main application window for the .tbm viewer."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Tensorbit .tbm Viewer")
        root.geometry("1100x720")
        root.minsize(800, 500)

        self.tbm_path: Optional[str] = None
        self.index: Optional[dict] = None
        self.tensors: list[dict] = []
        self.filtered_tensors: list[int] = []
        self.file_size: int = 0
        self._loaded = False

        self._build_ui()
        self._update_summary_panel()

    # ── UI Construction ──────────────────────

    def _build_ui(self):
        """Build all UI elements."""

        # -- menu bar --
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open .tbm file...", command=self._on_open,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.bind("<Control-o>", lambda _: self._on_open())
        self.root.bind("<Control-O>", lambda _: self._on_open())

        # -- root container: switches between welcome and main views --
        self._container = ttk.Frame(self.root)
        self._container.pack(fill=tk.BOTH, expand=True)

        # -- welcome screen (visible when no file is loaded) --
        self._welcome_frame = ttk.Frame(self._container)
        self._welcome_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        ttk.Label(self._welcome_frame,
                  text="Tensorbit .tbm Viewer",
                  font=("TkDefaultFont", 18, "bold")).pack(pady=(0, 8))
        ttk.Label(self._welcome_frame,
                  text="Open a .tbm model container file to inspect",
                  font=("TkDefaultFont", 11)).pack(pady=(0, 24))
        open_btn = ttk.Button(self._welcome_frame, text="Open .tbm file...",
                              command=self._on_open)
        open_btn.pack()
        ttk.Label(self._welcome_frame,
                  text="Ctrl+O",
                  foreground="gray", font=("TkDefaultFont", 9)).pack(pady=(8, 0))

        # -- main UI (hidden until file is loaded) --
        self._main_frame = ttk.Frame(self._container)

        # main paned window
        paned = ttk.PanedWindow(self._main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # -- left panel: summary + tensor list --
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)

        # summary bar
        self.summary_var = tk.StringVar(
            value="No file loaded.  File > Open .tbm file  (Ctrl+O)")
        summary_lbl = ttk.Label(left_frame, textvariable=self.summary_var,
                                relief=tk.SUNKEN, anchor=tk.W, padding=4)
        summary_lbl.pack(fill=tk.X, pady=(0, 4))

        # search bar
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(search_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filter())
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(search_frame, text="X", width=2,
                   command=lambda: self.search_var.set("")).pack(side=tk.RIGHT)

        # tensor treeview
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "shape", "weights", "nm")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                                  selectmode="browse")
        self.tree.heading("name", text="Tensor name")
        self.tree.heading("shape", text="Shape")
        self.tree.heading("weights", text="Weights")
        self.tree.heading("nm", text="N:M")
        self.tree.column("name", width=280, minwidth=120)
        self.tree.column("shape", width=100, minwidth=60, anchor=tk.CENTER)
        self.tree.column("weights", width=90, minwidth=60, anchor=tk.CENTER)
        self.tree.column("nm", width=60, minwidth=40, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", lambda _: self._on_tensor_select())

        # -- right panel: detail + preview --
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        # detail notebook
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # tab: metadata
        meta_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(meta_frame, text="Tensor Details")

        self.meta_text = tk.Text(meta_frame, wrap=tk.NONE, state=tk.DISABLED,
                                 height=12, font=("Courier", 10))
        meta_scroll = ttk.Scrollbar(meta_frame, orient=tk.VERTICAL,
                                    command=self.meta_text.yview)
        self.meta_text.configure(yscrollcommand=meta_scroll.set)
        self.meta_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        meta_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # tab: weight preview
        weight_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(weight_frame, text="Weight Preview")

        wctrl = ttk.Frame(weight_frame)
        wctrl.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(wctrl, text="Show first").pack(side=tk.LEFT)
        self.preview_count_var = tk.StringVar(value="128")
        pc_entry = ttk.Entry(wctrl, textvariable=self.preview_count_var,
                             width=8)
        pc_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(wctrl, text="values").pack(side=tk.LEFT)
        ttk.Button(wctrl, text="Refresh",
                   command=lambda: self._show_preview()).pack(side=tk.RIGHT)

        self.weight_text = tk.Text(weight_frame, wrap=tk.NONE,
                                   state=tk.DISABLED, font=("Courier", 10))
        w_scroll_y = ttk.Scrollbar(weight_frame, orient=tk.VERTICAL,
                                    command=self.weight_text.yview)
        w_scroll_x = ttk.Scrollbar(weight_frame, orient=tk.HORIZONTAL,
                                    command=self.weight_text.xview)
        self.weight_text.configure(yscrollcommand=w_scroll_y.set,
                                   xscrollcommand=w_scroll_x.set)
        self.weight_text.grid(row=2, column=0, sticky="nsew")
        w_scroll_y.grid(row=2, column=1, sticky="ns")
        w_scroll_x.grid(row=1, column=0, sticky="ew")
        weight_frame.rowconfigure(2, weight=1)
        weight_frame.columnconfigure(0, weight=1)

        # tab: mask preview
        mask_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(mask_frame, text="Mask Preview")

        self.mask_text = tk.Text(mask_frame, wrap=tk.NONE,
                                 state=tk.DISABLED, font=("Courier", 10))
        m_scroll_y = ttk.Scrollbar(mask_frame, orient=tk.VERTICAL,
                                    command=self.mask_text.yview)
        m_scroll_x = ttk.Scrollbar(mask_frame, orient=tk.HORIZONTAL,
                                    command=self.mask_text.xview)
        self.mask_text.configure(yscrollcommand=m_scroll_y.set,
                                 xscrollcommand=m_scroll_x.set)
        self.mask_text.grid(row=0, column=0, sticky="nsew")
        m_scroll_y.grid(row=0, column=1, sticky="ns")
        m_scroll_x.grid(row=1, column=0, sticky="ew")
        mask_frame.rowconfigure(0, weight=1)
        mask_frame.columnconfigure(0, weight=1)

        # tab: architecture diagram
        self._build_architecture_tab()

    def _build_architecture_tab(self):
        """Build the Architecture diagram tab."""
        arch_frame = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(arch_frame, text="Architecture")

        self.arch_canvas = tk.Canvas(arch_frame, bg="#1a1a2e", highlightthickness=0)
        h_scroll = ttk.Scrollbar(arch_frame, orient=tk.HORIZONTAL,
                                 command=self.arch_canvas.xview)
        v_scroll = ttk.Scrollbar(arch_frame, orient=tk.VERTICAL,
                                 command=self.arch_canvas.yview)
        self.arch_canvas.configure(xscrollcommand=h_scroll.set,
                                    yscrollcommand=v_scroll.set)
        self.arch_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        arch_frame.rowconfigure(0, weight=1)
        arch_frame.columnconfigure(0, weight=1)

        self.arch_canvas.bind("<Enter>", self._on_arch_canvas_enter)
        self.arch_canvas.bind("<MouseWheel>", self._on_arch_mousewheel)
        self.arch_canvas.bind("<Button-4>", self._on_arch_mousewheel_linux)
        self.arch_canvas.bind("<Button-5>", self._on_arch_mousewheel_linux)
        self.arch_canvas.bind("<Motion>", self._on_arch_motion)
        self.arch_canvas.bind("<Leave>", self._on_arch_canvas_leave)
        self._arch_data = None
        self._arch_expanded = {}
        self._arch_tag_colors = {}
        self._current_hover_tag = None

    # ── View switching ──────────────────────

    def _show_welcome(self):
        self._welcome_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self._main_frame.pack_forget()

    def _show_main(self):
        self._welcome_frame.place_forget()
        self._main_frame.pack(fill=tk.BOTH, expand=True)

    # ── Actions ──────────────────────────────

    def _on_open(self):
        """Open a .tbm file via dialog."""
        path = filedialog.askopenfilename(
            title="Open .tbm model container",
            filetypes=[("Tensorbit Model files", "*.tbm"), ("All files", "*.*")])
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path: str):
        """Load and parse a .tbm file."""
        self.tbm_path = path
        self.index = None
        self.tensors.clear()
        self.filtered_tensors.clear()
        self.file_size = 0

        self.tree.delete(*self.tree.get_children())
        self._clear_detail()
        self._clear_preview()

        index = parse_tbm(path)
        if index is None:
            self._show_welcome()
            messagebox.showerror("Parse Error",
                                 "Failed to parse .tbm file.\n"
                                 "Not a valid .tbm container or file is corrupted.")
            self._update_summary_panel()
            return

        self.index = index
        self.tensors = index.get("tensors", [])
        self.file_size = os.path.getsize(path)

        # populate tree
        for i, t in enumerate(self.tensors):
            name = t.get("name", f"tensor_{i}")
            shape = t.get("shape", [])
            shape_str = "×".join(str(d) for d in shape) if shape else "?"
            nw = t.get("num_weights", 0)
            nw_str = f"{nw:,}"
            n_val = t.get("nm_n", "?")
            m_val = t.get("nm_m", "?")
            nm_str = f"{n_val}:{m_val}"
            self.tree.insert("", tk.END, iid=str(i),
                             values=(name, shape_str, nw_str, nm_str))

        self.filtered_tensors = list(range(len(self.tensors)))
        self._show_main()
        self._update_summary_panel()
        self._render_diagram()

    def _apply_filter(self):
        """Filter the treeview by search text."""
        query = self.search_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        self.filtered_tensors.clear()

        for i, t in enumerate(self.tensors):
            name = t.get("name", f"tensor_{i}")
            if query and query not in name.lower():
                continue
            self.filtered_tensors.append(i)
            shape = t.get("shape", [])
            shape_str = "×".join(str(d) for d in shape) if shape else "?"
            nw = t.get("num_weights", 0)
            nw_str = f"{nw:,}"
            n_val = t.get("nm_n", "?")
            m_val = t.get("nm_m", "?")
            nm_str = f"{n_val}:{m_val}"
            self.tree.insert("", tk.END, iid=str(i),
                             values=(name, shape_str, nw_str, nm_str))

    def _on_tensor_select(self):
        """Handle tensor selection in treeview."""
        sel = self.tree.selection()
        if not sel:
            self._clear_detail()
            self._clear_preview()
            return

        idx = int(sel[0])
        if idx >= len(self.tensors):
            return

        tensor = self.tensors[idx]
        self._show_detail(tensor)
        self._show_preview(tensor)

    def _show_detail(self, tensor: dict):
        """Populate the metadata text widget."""
        name = tensor.get("name", "?")
        offset = tensor.get("offset", 0)
        shape = tensor.get("shape", [])
        shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
        nw = tensor.get("num_weights", 0)
        nmb = tensor.get("num_mask_bytes", 0)
        dtype = tensor.get("dtype", "fp32")
        nm_n = tensor.get("nm_n", "?")
        nm_m = tensor.get("nm_m", "?")

        # compute data sizes
        elem_size = DTYPE_SIZES.get(dtype, 4)
        weight_mb = nw * elem_size / (1024 * 1024)
        mask_kb = nmb / 1024.0
        total_kb = (TB_HEADER_SIZE + nw * elem_size + nmb) / 1024.0
        sparsity = 1.0 - (nm_n / nm_m) if isinstance(nm_n, int) and isinstance(nm_m, int) and nm_m > 0 else 0.0

        text = (
            f"Name:          {name}\n"
            f"Shape:         {shape_str}\n"
            f"Dtype:         {dtype}\n"
            f"N:M sparsity:  {nm_n}:{nm_m} ({sparsity:.0%} sparse)\n"
            f"Num weights:   {nw:,}\n"
            f"Num mask bytes:{nmb:,}\n"
            f"Header offset: {offset:,}  (0x{offset:X})\n"
            f"Data offset:   {offset + TB_HEADER_SIZE:,}  "
            f"(0x{offset + TB_HEADER_SIZE:X})\n"
            f"Weight data:   {weight_mb:.2f} MB\n"
            f"Mask data:     {mask_kb:.2f} KB\n"
            f"Total on disk: {total_kb:.2f} KB\n"
        )

        if self.file_size > 0:
            blobs_end = self.file_size - 4
            if self.index:
                idx_len = len(json.dumps(self.index, separators=(",", ":")))
                blobs_end = self.file_size - 4 - idx_len

            text += (
                f"\n"
                f"File position: {offset:,} .. {int(offset + total_kb * 1024):,}"
                f"  ({100.0 * offset / max(self.file_size, 1):.1f}%)\n"
            )

        self.meta_text.configure(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert("1.0", text)
        self.meta_text.configure(state=tk.DISABLED)

    def _show_preview(self, tensor: Optional[dict] = None):
        """Populate weight and mask preview tabs."""
        if tensor is None:
            # called from refresh button
            sel = self.tree.selection()
            if not sel:
                return
            idx = int(sel[0])
            if idx >= len(self.tensors):
                return
            tensor = self.tensors[idx]

        if not self.tbm_path:
            return

        # --- weight preview ---
        try:
            count = int(self.preview_count_var.get())
        except ValueError:
            count = 128
        count = max(1, min(count, 1024))

        offset = tensor.get("offset", 0)
        nw = tensor.get("num_weights", 0)
        dtype = tensor.get("dtype", "fp32")
        nmb = tensor.get("num_mask_bytes", 0)

        vals = preview_tensor(self.tbm_path, offset, nw, dtype, count)

        self.weight_text.configure(state=tk.NORMAL)
        self.weight_text.delete("1.0", tk.END)

        if vals is None:
            self.weight_text.insert("1.0", "(failed to read weight data)")
        else:
            lines = []
            for i, v in enumerate(vals):
                if isinstance(v, float):
                    fstr = f"{v: .6e}"
                else:
                    fstr = str(v)
                lines.append(f"[{i:4d}] {fstr}")
            if nw > count:
                lines.append(f"... ({nw - count:,} more values not shown)")
            self.weight_text.insert("1.0", "\n".join(lines))

        self.weight_text.configure(state=tk.DISABLED)

        # --- mask preview ---
        mask_vals = preview_masks(self.tbm_path, offset, nmb, count=256)

        self.mask_text.configure(state=tk.NORMAL)
        self.mask_text.delete("1.0", tk.END)

        if mask_vals is None:
            self.mask_text.insert("1.0", "(failed to read mask data)")
        elif len(mask_vals) == 0:
            self.mask_text.insert("1.0", "(no mask data)")
        else:
            lines = []
            for i in range(0, len(mask_vals), 16):
                chunk = mask_vals[i:i + 16]
                hex_part = " ".join(f"{b:02X}" for b in chunk)
                lines.append(f"{i:6d}:  {hex_part}")
            if nmb > len(mask_vals):
                lines.append(f"... ({nmb - len(mask_vals):,} more bytes not shown)")
            self.mask_text.insert("1.0", "\n".join(lines))

        self.mask_text.configure(state=tk.DISABLED)

    def _clear_detail(self):
        """Clear the metadata tab."""
        self.meta_text.configure(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert("1.0", "(select a tensor from the list)")
        self.meta_text.configure(state=tk.DISABLED)

    def _clear_preview(self):
        """Clear weight and mask preview tabs."""
        for tw in (self.weight_text, self.mask_text):
            tw.configure(state=tk.NORMAL)
            tw.delete("1.0", tk.END)
            tw.insert("1.0", "(select a tensor from the list)")
            tw.configure(state=tk.DISABLED)

    def _update_summary_panel(self):
        """Update the summary bar with file info."""
        if self.tbm_path is None:
            self.summary_var.set(
                "No file loaded.  File > Open .tbm file  (Ctrl+O)")
        else:
            arch = self.index.get("architecture", "?") if self.index else "?"
            num = len(self.tensors)
            size_mb = self.file_size / (1024 * 1024)
            total_wts = sum(
                t.get("num_weights", 0) for t in self.tensors)
            fname = os.path.basename(self.tbm_path)
            self.summary_var.set(
                f"{fname}  |  {num} tensors  |  "
                f"architecture: {arch}  |  "
                f"{size_mb:.0f} MB  |  {total_wts:,} total weights")

    # ── Architecture diagram ─────────────────

    def _infer_architecture(self):
        """Parse tensor names into a structured layer graph.

        Returns a dict with:
          - 'embed':     list of embedding tensors
          - 'layers':    list of dicts, each with 'attn', 'mlp', 'norms' lists
          - 'final_norm': list of final normalization tensors
          - 'lm_head':   list of lm_head tensors
          - 'other':     any tensors that don't match known patterns
        """
        layers = []
        special = {'embed': [], 'final_norm': [], 'lm_head': [], 'other': []}

        for t in self.tensors:
            name = t.get('name', '')
            if not name:
                special['other'].append(t)
                continue

            if 'embed' in name.lower():
                special['embed'].append(t)
            elif name == 'model.norm.weight' or name.endswith('.norm.weight'):
                special['final_norm'].append(t)
            elif 'lm_head' in name:
                special['lm_head'].append(t)
            elif 'layers.' in name:
                m = re.search(r'layers\.(\d+)', name)
                if m:
                    idx = int(m.group(1))
                    while len(layers) <= idx:
                        layers.append({'index': len(layers),
                                       'attn': [], 'mlp': [], 'norms': []})
                    if 'self_attn' in name or 'attention' in name:
                        layers[idx]['attn'].append(t)
                    elif 'mlp' in name or 'feed_forward' in name:
                        layers[idx]['mlp'].append(t)
                    elif 'norm' in name.lower() or 'layernorm' in name.lower():
                        layers[idx]['norms'].append(t)
                    else:
                        layers[idx]['attn'].append(t)
                else:
                    special['other'].append(t)
            else:
                special['other'].append(t)

        # Sort sub-tensors within each group by name for consistent layout
        for layer in layers:
            for key in ('attn', 'mlp', 'norms'):
                layer[key].sort(key=lambda x: x.get('name', ''))

        for key in ('embed', 'final_norm', 'lm_head', 'other'):
            special[key].sort(key=lambda x: x.get('name', ''))

        return {'layers': layers, 'special': special}

    _ARCH_COLORS = {
        'embed':       '#4A90D9',
        'q_proj':      '#50C878',
        'k_proj':      '#3CB371',
        'v_proj':      '#2E8B57',
        'o_proj':      '#66CDAA',
        'attn':        '#50C878',
        'gate_proj':   '#FF8C42',
        'up_proj':     '#FF7B25',
        'down_proj':   '#E06930',
        'mlp':         '#FF8C42',
        'norm':        '#9B9B9B',
        'final_norm':  '#7B8D9E',
        'lm_head':     '#E74C3C',
        'other':       '#888888',
    }

    _ARCH_GROUP_LABELS = {
        'embed':       'Embedding',
        'attn':        'Self-Attention',
        'mlp':         'MLP (FFN)',
        'norms':       'LayerNorm',
        'final_norm':  'Final Norm',
        'lm_head':     'LM Head',
        'other':       'Other',
    }

    @staticmethod
    def _role_of_tensor(name: str, group: str) -> str:
        """Return a short role label for a tensor within its group."""
        for kw in ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                   'gate_proj', 'up_proj', 'down_proj',
                    'embed', 'lm_head', 'norm'):
            if kw in name:
                return kw
        return group

    @staticmethod
    def _color_of(name: str, group: str) -> str:
        role = TbmViewerApp._role_of_tensor(name, group)
        return TbmViewerApp._ARCH_COLORS.get(role,
                TbmViewerApp._ARCH_COLORS.get(group, '#888888'))

    @staticmethod
    def _short_name(name: str) -> str:
        """Extract a compact label from a tensor name."""
        name = name.replace('model.', '').replace('.weight', '')
        name = re.sub(r'layers\.\d+\.', '', name)
        return name

    def _render_diagram(self):
        """Draw the architecture diagram on the canvas."""
        canvas = self.arch_canvas
        canvas.delete('all')
        self._arch_tag_colors.clear()
        self._current_hover_tag = None

        if not self.tensors:
            canvas.create_text(400, 100, text="(no tensors loaded)",
                               fill="#888888", font=("TkDefaultFont", 14))
            canvas.configure(scrollregion=(0, 0, 800, 200))
            return

        arch = self._infer_architecture()
        self._arch_data = arch

        layers = arch['layers']
        special = arch['special']
        num_layers = len(layers)

        if num_layers == 0:
            canvas.create_text(400, 100, text="(no layer structure detected)",
                               fill="#888888", font=("TkDefaultFont", 14))
            canvas.configure(scrollregion=(0, 0, 800, 200))
            return

        # Layout constants
        W = 720
        X0 = 40
        X_MID = X0 + W // 2
        BOX_W = 130
        BOX_H = 28
        GAP_X = 10
        GAP_Y = 14
        LAYER_PAD_Y = 10
        ARROW_COLOR = "#AAAAAA"
        TEXT_COLOR = "#E0E0E0"
        HEADER_COLOR = "#333355"
        HEADER_H = 24

        y = 20
        item_tags = {}

        # --- Legend ---
        legend_x = X0 + W - 200
        ly = y
        canvas.create_text(legend_x + 60, ly, text="Legend",
                           fill="#BBBBBB", font=("TkDefaultFont", 10, "bold"),
                           anchor=tk.N)
        ly += 16
        legend_items = [
            ('Attention', '#50C878'), ('MLP', '#FF8C42'),
            ('Norm', '#9B9B9B'), ('Embed', '#4A90D9'),
            ('LM Head', '#E74C3C'),
        ]
        for ltext, lcolor in legend_items:
            canvas.create_rectangle(legend_x, ly, legend_x + 14, ly + 10,
                                    fill=lcolor, outline=lcolor)
            canvas.create_text(legend_x + 18, ly + 5, text=ltext,
                               fill="#BBBBBB", font=("TkDefaultFont", 9),
                               anchor=tk.W)
            ly += 14

        # --- Embedding ---
        y += 4
        if special['embed']:
            y = self._draw_group_header(canvas, X0, y, W,
                                         TbmViewerApp._ARCH_GROUP_LABELS['embed'],
                                         '#4A90D9')
            for t in special['embed']:
                y = self._draw_tensor_box(canvas, X_MID - BOX_W // 2, y,
                                           BOX_W, BOX_H, t, 'embed')
                y += GAP_Y
            y += 8
            self._draw_arrow(canvas, X_MID, y - 4, X_MID, y + 10)
            y += 14

        # --- Layer blocks ---
        total_layers = len(layers)
        for li, ldata in enumerate(layers):
            lnum = ldata['index']
            num_sub = (len(ldata['norms']) + len(ldata['attn'])
                       + len(ldata['mlp']))
            header_text = f"Layer {lnum}  ({num_sub} tensors)"
            expanded = self._arch_expanded.get(lnum, num_sub <= 12)

            y = self._draw_group_header(canvas, X0, y, W, header_text,
                                         HEADER_COLOR, lnum, expanded)

            if expanded:
                # Norms first
                for tn in ldata['norms']:
                    y = self._draw_tensor_box(canvas, X_MID - BOX_W // 2, y,
                                               BOX_W, BOX_H, tn, 'norms')
                    y += GAP_Y
                y -= GAP_Y

                # Attention group
                if ldata['attn']:
                    y += 4
                    mx = X_MID - (len(ldata['attn']) * (BOX_W + GAP_X) - GAP_X) // 2
                    for ti, ta in enumerate(ldata['attn']):
                        bx = mx + ti * (BOX_W + GAP_X)
                        y2 = self._draw_tensor_box(canvas, bx, y, BOX_W, BOX_H,
                                                    ta, 'attn')
                        if ti < len(ldata['attn']) - 1:
                            self._draw_arrow(canvas,
                                             bx + BOX_W // 2, y2,
                                             bx + BOX_W // 2 + BOX_W + GAP_X, y2)
                    y = y2 + GAP_Y

                # MLP group
                if ldata['mlp']:
                    y += 4
                    mx = X_MID - (len(ldata['mlp']) * (BOX_W + GAP_X) - GAP_X) // 2
                    for ti, tm in enumerate(ldata['mlp']):
                        bx = mx + ti * (BOX_W + GAP_X)
                        y = self._draw_tensor_box(canvas, bx, y, BOX_W, BOX_H,
                                                   tm, 'mlp')
                        if ti < len(ldata['mlp']) - 1:
                            self._draw_arrow(canvas,
                                             bx + BOX_W // 2, y + BOX_H,
                                             bx + BOX_W // 2 + BOX_W + GAP_X, y + BOX_H)
                    y += GAP_Y

            y += LAYER_PAD_Y

            if li < total_layers - 1:
                self._draw_arrow(canvas, X_MID, y, X_MID, y + 16)
                y += 20

        # --- Final Norm ---
        if special['final_norm']:
            y += 4
            y = self._draw_group_header(canvas, X0, y, W,
                                         TbmViewerApp._ARCH_GROUP_LABELS['final_norm'],
                                         '#7B8D9E')
            for t in special['final_norm']:
                y = self._draw_tensor_box(canvas, X_MID - BOX_W // 2, y,
                                           BOX_W, BOX_H, t, 'final_norm')
                y += GAP_Y
            y += 4
            self._draw_arrow(canvas, X_MID, y, X_MID, y + 14)
            y += 18

        # --- LM Head ---
        if special['lm_head']:
            y = self._draw_group_header(canvas, X0, y, W,
                                         TbmViewerApp._ARCH_GROUP_LABELS['lm_head'],
                                         '#E74C3C')
            for t in special['lm_head']:
                y = self._draw_tensor_box(canvas, X_MID - BOX_W // 2, y,
                                           BOX_W, BOX_H, t, 'lm_head')
                y += GAP_Y

        # --- Others ---
        if special['other']:
            y += 12
            y = self._draw_group_header(canvas, X0, y, W, 'Other Tensors',
                                         '#555555')
            for t in special['other']:
                y = self._draw_tensor_box(canvas, X_MID - BOX_W // 2, y,
                                           BOX_W, BOX_H, t, 'other')
                y += GAP_Y

        y += 30
        model_name = self.index.get('architecture', 'model') if self.index else 'model'
        canvas.create_text(X_MID, y,
                           text=f"{model_name} — {num_layers} layers, "
                                f"{len(self.tensors)} tensors",
                           fill="#666688", font=("TkDefaultFont", 10, "italic"))
        y += 30

        canvas.configure(scrollregion=(0, 0, max(X0 + W, 820), y))

    def _draw_group_header(self, canvas, x, y, w, text, color,
                            lnum=None, expanded=True):
        """Draw a collapsible group header. Returns new y after the header."""
        h = 24
        is_layer = lnum is not None
        tag = f"ghdr_{lnum}" if is_layer else f"ghdr_{text}"

        box_id = canvas.create_rectangle(x, y, x + w, y + h,
                                          fill=color, outline=color,
                                          stipple='gray25' if not expanded else '',
                                          tags=(tag,))
        toggle_id = canvas.create_text(x + h // 2, y + h // 2,
                                        text='-' if expanded else '+',
                                        fill="white",
                                        font=("TkDefaultFont", 11, "bold"),
                                        tags=(tag,))
        canvas.create_text(x + h + 8, y + h // 2, text=text,
                           fill="white",
                           font=("TkDefaultFont", 10, "bold"),
                           anchor=tk.W, tags=(tag,))

        if is_layer:
            self._arch_tag_colors[tag] = color
            canvas.tag_bind(tag, "<Button-1>",
                            lambda e, ln=lnum: self._toggle_layer(ln))
        return y + h + 6

    def _toggle_layer(self, lnum: int):
        self._arch_expanded[lnum] = not self._arch_expanded.get(lnum, True)
        self._render_diagram()

    def _draw_tensor_box(self, canvas, x, y, w, h, tensor, group):
        """Draw a single tensor rectangle on the canvas. Returns new y."""
        name = tensor.get('name', '?')
        short = TbmViewerApp._short_name(name)
        color = TbmViewerApp._color_of(name, group)
        shape = tensor.get('shape', [])
        nw = tensor.get('num_weights', 0)
        shape_str = chr(0x00D7).join(str(d) for d in shape) if shape else '?'
        label = short[:18] + ('..' if len(short) > 18 else '')

        idx = self.tensors.index(tensor) if tensor in self.tensors else -1
        tag = f"tbox_{idx}"
        hover_tag = f"hover_tbox_{idx}"

        canvas.create_rectangle(x, y, x + w, y + h,
                                 fill=color, outline=color,
                                 tags=(tag,))
        canvas.create_text(x + w // 2, y + h // 2,
                           text=label, fill="white",
                           font=("TkDefaultFont", 9, "bold"),
                           tags=(tag,))
        canvas.create_text(x + w // 2, y - 2,
                           text=f"{shape_str}  {nw:,}",
                           fill="#999999",
                           font=("TkDefaultFont", 7), anchor=tk.S,
                           tags=(tag,))

        if idx >= 0:
            self._arch_tag_colors[tag] = color
            canvas.tag_bind(tag, "<Button-1>",
                            lambda e, i=idx: self._show_heatmap(i))
        return y + h

    def _draw_arrow(self, canvas, x1, y1, x2, y2):
        """Draw a downward arrow."""
        canvas.create_line(x1, y1, x2, y2 - 5, fill="#666688", width=2,
                           arrow=tk.LAST, arrowshape=(8, 10, 4))

    def _on_arch_canvas_enter(self, event):
        """Focus the canvas so mousewheel events are captured."""
        self.arch_canvas.focus_set()

    def _on_arch_mousewheel(self, event):
        """Mouse wheel scrolling (Windows/macOS)."""
        self.arch_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_arch_mousewheel_linux(self, event):
        """Mouse wheel scrolling (Linux, X11)."""
        if event.num == 4:
            self.arch_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.arch_canvas.yview_scroll(1, "units")

    def _on_arch_canvas_leave(self, event):
        """Unhover when mouse leaves the canvas entirely."""
        self._set_hover_tag(None)

    def _on_arch_motion(self, event):
        """Track which tag (tbox_ or ghdr_) the mouse is over.  Only re-
        highlight when the tag actually changes — prevents blinking when
        cursor moves between items that share the same tag."""
        canvas = self.arch_canvas
        cx = canvas.canvasx(event.x)
        cy = canvas.canvasy(event.y)
        items = canvas.find_overlapping(cx - 2, cy - 2, cx + 2, cy + 2)
        found = None
        for iid in items:
            for t in canvas.gettags(iid):
                if t.startswith('tbox_') or t.startswith('ghdr_'):
                    found = t
                    break
            if found:
                break
        self._set_hover_tag(found)

    def _set_hover_tag(self, new_tag):
        """Switch hover highlight to new_tag (None = unhover)."""
        if new_tag == self._current_hover_tag:
            return
        canvas = self.arch_canvas

        if (self._current_hover_tag and
                self._current_hover_tag in self._arch_tag_colors):
            orig = self._arch_tag_colors[self._current_hover_tag]
            self._apply_tag_color(canvas, self._current_hover_tag,
                                   orig, orig)

        if new_tag and new_tag in self._arch_tag_colors:
            orig = self._arch_tag_colors[new_tag]
            r = int(orig[1:3], 16)
            g = int(orig[3:5], 16)
            b = int(orig[5:7], 16)
            r = min(255, r + 45)
            g = min(255, g + 45)
            b = min(255, b + 45)
            hl = f"#{r:02x}{g:02x}{b:02x}"
            self._apply_tag_color(canvas, new_tag, hl, hl)

        self._current_hover_tag = new_tag

    @staticmethod
    def _apply_tag_color(canvas, tag, fill, outline):
        """Apply fill/outline to every rectangle item with the given tag."""
        for iid in canvas.find_withtag(tag):
            if canvas.type(iid) == 'rectangle':
                canvas.itemconfigure(iid, fill=fill, outline=outline)

    def _show_heatmap(self, tensor_idx: int):
        """Show a heatmap popup of the first K*K weights of a tensor."""
        if tensor_idx < 0 or tensor_idx >= len(self.tensors):
            return
        if not self.tbm_path:
            return

        tensor = self.tensors[tensor_idx]
        nw = tensor.get('num_weights', 0)
        dtype = tensor.get('dtype', 'fp32')
        offset = tensor.get('offset', 0)

        side = min(int(math.sqrt(min(nw, 4096))), 48)
        if side < 2:
            return
        count = side * side

        vals = preview_tensor(self.tbm_path, offset, nw, dtype, count)
        if vals is None or len(vals) < 2:
            return

        max_abs = max(abs(v) for v in vals if math.isfinite(v))
        if max_abs == 0:
            max_abs = 1.0

        popup = tk.Toplevel(self.root)
        popup.title(f"Heatmap — {TbmViewerApp._short_name(tensor.get('name', '?'))}")
        popup.resizable(False, False)

        CELL = 12
        pad = 8
        canvas_w = side * CELL + pad * 2
        canvas_h = side * CELL + pad * 2 + 24

        hc = tk.Canvas(popup, width=canvas_w, height=canvas_h,
                       bg="#0d0d1a", highlightthickness=0)
        hc.pack()

        for row in range(side):
            for col in range(side):
                v = vals[row * side + col]
                if not math.isfinite(v):
                    r, g, b = 100, 100, 100
                else:
                    t = v / max_abs
                    t = max(-1.0, min(1.0, t))
                    if t >= 0:
                        r = int(30 + 225 * t)
                        g = int(30 + 60 * (1 - t))
                        b = int(60 + 60 * (1 - t))
                    else:
                        t = -t
                        r = int(60 + 60 * (1 - t))
                        g = int(30 + 60 * (1 - t))
                        b = int(30 + 225 * t)

                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                px = pad + col * CELL
                py = pad + row * CELL

                hc.create_rectangle(px, py, px + CELL - 1, py + CELL - 1,
                                    fill=hex_color, outline=hex_color)

        name = TbmViewerApp._short_name(tensor.get('name', '?'))
        shape = tensor.get('shape', [])
        shape_str = '×'.join(str(d) for d in shape) if shape else '?'
        sp = 0.0
        nm_n = tensor.get('nm_n')
        nm_m = tensor.get('nm_m')
        if isinstance(nm_n, (int, float)) and isinstance(nm_m, (int, float)) and nm_m > 0:
            sp = 1.0 - (nm_n / nm_m)

        hc.create_text(pad + (side * CELL) // 2, canvas_h - 6,
                       text=f"{name}  {shape_str}  {sp:.0%} sparse  "
                            f"[max abs = {max_abs:.4f}]",
                       fill="#888888", font=("TkDefaultFont", 8))

        # Explanation
        expl_frame = ttk.Frame(popup, padding=(8, 4, 8, 8))
        expl_frame.pack(fill=tk.X)

        mean_val = sum(v for v in vals if math.isfinite(v)) / max(len(vals), 1)
        non_zero = sum(1 for v in vals if math.isfinite(v) and abs(v) > 1e-8)
        zero_pct = (1.0 - non_zero / max(len(vals), 1)) * 100

        expl_lines = [
            f"showing first {side}\u00D7{side} = {count} weight values",
            f"blue = negative   \u00B7   red = positive   \u00B7   gray = NaN/Inf",
            f"color saturates at \u00B1{max_abs:.4f} (the max absolute value in this slice)",
        ]
        if count >= 4:
            expl_lines.append(
                f"mean = {mean_val:+.4f}   \u00B7   "
                f"{zero_pct:.0f}% zeros ({non_zero}/{count} non-zero)")
        full = "\n".join(expl_lines)
        ttk.Label(expl_frame, text=full, justify=tk.LEFT,
                   font=("TkDefaultFont", 8),
                   foreground="#aaaaaa", background="#1a1a1a").pack(
                       fill=tk.X, anchor=tk.W)
        popup.configure(background="#1a1a1a")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main():
    root = tk.Tk()
    app = TbmViewerApp(root)

    # accept file path as command-line argument
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path) and path.lower().endswith(".tbm"):
            app._load_file(path)

    root.mainloop()


if __name__ == "__main__":
    main()
