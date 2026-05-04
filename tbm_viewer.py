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
        self._container = ttk.Frame(root)
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
        wctrl.pack(fill=tk.X, pady=(0, 4))
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
        self.weight_text.grid(row=1, column=0, sticky="nsew")
        w_scroll_y.grid(row=1, column=1, sticky="ns")
        w_scroll_x.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        weight_frame.rowconfigure(1, weight=1)
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
