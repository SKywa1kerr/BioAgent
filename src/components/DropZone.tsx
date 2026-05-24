import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { UploadCloud } from "lucide-react";
import { useReducedMotion } from "framer-motion";
import { useToasts } from "./ui/ToastProvider";
import { pairAb1Gb } from "../lib/files/pairAb1Gb.js";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import "./DropZone.css";

interface DropZoneProps {
  language: AppLanguage;
  /** Children render inside the dropzone container so the overlay sits on top. */
  children: ReactNode;
  /** Triggered when the user drops a folder (or folders) instead of files —
   *  App opens the register-dataset dialog with the path pre-filled. */
  onRequestImport?: (prefill: { ab1Dir?: string; gbDir?: string }) => void;
  /** Triggered when the dropped folder contains several dataset-shaped
   *  subfolders — App shows the chooser dialog instead of the import form. */
  onRequestChoose?: (candidates: Array<{ name: string; path: string; layout: "subdirs" | "flat" }>) => void;
  /** Triggered when the dropped files aren't analyzable on their own
   *  (PDFs, single .ab1, mixed extensions). Host prefills the composer
   *  with a "please read these" prompt so the agent can call read_pdf /
   *  read_sequence_file. */
  onRequestComposerPrefill?: (text: string) => void;
}

interface DroppedFilesResponse {
  ok: boolean;
  error?: string;
  result?: { sample_count?: number } & Record<string, unknown>;
  pairs?: number;
  unpairedAb1?: string[];
  unpairedGb?: string[];
}

const AB1_RE = /\.ab1$/i;
const GB_RE = /\.(gb|gbk)$/i;
const PDF_RE = /\.pdf$/i;

function dragHasFiles(event: DragEvent): boolean {
  const types = event.dataTransfer?.types;
  if (!types) return false;
  for (let i = 0; i < types.length; i += 1) {
    if (types[i] === "Files") return true;
  }
  return false;
}

/**
 * Renderer-side drop zone. Listens to drag/drop events on `window`, splits
 * dropped paths into AB1 / GB lists, runs the pure pairing helper for an
 * instant local check, and forwards the lists to the main process via the
 * `analyze-dropped-files` IPC channel. Feedback is delivered through toasts.
 */
export function DropZone({ language, children, onRequestImport, onRequestChoose, onRequestComposerPrefill }: DropZoneProps): JSX.Element {
  const [dragActive, setDragActive] = useState(false);
  const dragCounterRef = useRef(0);
  const toasts = useToasts();
  const reduceMotion = useReducedMotion() ?? false;

  // Latest language captured in a ref so the (window-scoped) drop handler
  // always toasts in the current language without needing a fresh effect.
  const languageRef = useRef(language);
  useEffect(() => {
    languageRef.current = language;
  }, [language]);

  const resetDrag = useCallback(() => {
    dragCounterRef.current = 0;
    setDragActive(false);
  }, []);

  const processDrop = useCallback(
    async (fileList: FileList) => {
      const lang = languageRef.current;
      const ab1Paths: string[] = [];
      const gbPaths: string[] = [];
      const allPaths: string[] = [];

      for (let i = 0; i < fileList.length; i += 1) {
        const file = fileList.item(i);
        if (!file) continue;
        // Electron 32+ removed File.path. The preload bridge exposes
        // webUtils.getPathForFile as electronAPI.getDroppedFilePath; older
        // builds may still surface file.path so we fall back to it.
        let path = "";
        const bridge = window.electronAPI as { getDroppedFilePath?: (f: File) => string };
        if (typeof bridge.getDroppedFilePath === "function") {
          path = bridge.getDroppedFilePath(file) || "";
        }
        if (!path) {
          const legacy = (file as File & { path?: unknown }).path;
          if (typeof legacy === "string") path = legacy;
        }
        if (path.length === 0) continue;
        allPaths.push(path);
        if (AB1_RE.test(path)) ab1Paths.push(path);
        else if (GB_RE.test(path)) gbPaths.push(path);
      }
      // eslint-disable-next-line no-console
      console.log("[DropZone] drop received:", { allPaths, ab1Paths, gbPaths, fileCount: fileList.length });

      // Always probe for folders in the drop set. Previously we only looked
      // at folders when NO file extensions matched, which silently lost
      // folders in mixed file+folder drops. Now we run inspect-dropped-paths
      // unconditionally and route to the folder/dataset path when any of the
      // dropped items is a directory.
      let droppedDirs: string[] = [];
      if (allPaths.length > 0) {
        try {
          const inspectAll = await window.electronAPI.invoke("inspect-dropped-paths", allPaths);
          droppedDirs = Array.isArray(inspectAll?.dirs) ? inspectAll.dirs : [];
          // eslint-disable-next-line no-console
          console.log("[DropZone] inspect-dropped-paths →", { allPaths, droppedDirs });
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error("[DropZone] inspect-dropped-paths threw:", err);
        }
      }

      // If the user dropped any folders, treat that as the strong signal of
      // intent ("import this folder as a dataset") and route through the
      // folder-handling path. Loose files dropped alongside are ignored,
      // which is the right call: mixed drops are almost always accidents.
      if (droppedDirs.length > 0 && (onRequestImport || onRequestChoose)) {
        try {
          if (droppedDirs.length === 1) {
            const folder = droppedDirs[0]!;
            const inspected = await window.electronAPI.invoke("inspect-dataset-folder", folder);
            // eslint-disable-next-line no-console
            console.log("[DropZone] inspect-dataset-folder →", { folder, inspected });
            if (inspected?.ok && inspected.layout === "multi" && Array.isArray(inspected.candidates) && inspected.candidates.length > 0) {
              // eslint-disable-next-line no-console
              console.log("[DropZone] routing to chooser with", inspected.candidates.length, "candidates");
              if (onRequestChoose) {
                onRequestChoose(inspected.candidates);
              } else {
                onRequestImport?.({ ab1Dir: folder, gbDir: folder });
              }
              return;
            }
            if (inspected?.ok) {
              onRequestImport?.({ ab1Dir: inspected.ab1Dir, gbDir: inspected.gbDir });
            } else {
              onRequestImport?.({ ab1Dir: folder, gbDir: folder });
            }
            return;
          }
          if (droppedDirs.length === 2) {
            const [d0, d1] = droppedDirs as [string, string];
            const lower0 = d0.toLowerCase();
            const lower1 = d1.toLowerCase();
            const looksAb1 = (s: string) => /ab1/.test(s);
            const looksGb = (s: string) => /\bgb\b|gbk/.test(s);
            let ab1Dir = d0;
            let gbDir = d1;
            if (looksAb1(lower1) || looksGb(lower0)) {
              ab1Dir = d1;
              gbDir = d0;
            }
            onRequestImport?.({ ab1Dir, gbDir });
            return;
          }
          // >2 dirs: too ambiguous, surface only the first two and let the
          // user fix paths in the dialog.
          onRequestImport?.({ ab1Dir: droppedDirs[0], gbDir: droppedDirs[1] });
          return;
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error("[DropZone] folder routing threw, falling through to file pairing:", err);
        }
      }

      // Single-file paths the analyze pipeline can't use: PDFs, lone ab1
      // or gb files without a partner, or anything we don't recognise.
      // Route into the composer-prefill flow so the agent can read them
      // via read_pdf / read_sequence_file. This mirrors the 📎 picker.
      const pdfPaths = allPaths.filter((p) => PDF_RE.test(p));
      const knownExt = (p: string) => AB1_RE.test(p) || GB_RE.test(p) || PDF_RE.test(p);
      const otherFiles = allPaths.filter((p) => !knownExt(p));

      if (pdfPaths.length > 0 && onRequestComposerPrefill) {
        const intro = lang === "zh"
          ? "请帮我读一下并总结这些 PDF："
          : "Please read and summarize these PDFs:";
        onRequestComposerPrefill(`${intro}\n${pdfPaths.map((p) => `- ${p}`).join("\n")}`);
        // If only PDFs were dropped, we're done. Otherwise fall through
        // so the .ab1/.gb pairing still runs alongside.
        if (ab1Paths.length === 0 && gbPaths.length === 0) return;
      }

      const { pairs, unpairedAb1, unpairedGb } = pairAb1Gb(ab1Paths, gbPaths);

      if (pairs.length === 0) {
        // Before the loud "no pairs" toast, try the gentle composer-
        // prefill fallback for any single .ab1 / .gb / unknown files so
        // the user isn't stuck.
        if (onRequestComposerPrefill && (ab1Paths.length > 0 || gbPaths.length > 0 || otherFiles.length > 0)) {
          const intro = lang === "zh"
            ? "请帮我看一下这些文件："
            : "Please take a look at these files:";
          const all = [...ab1Paths, ...gbPaths, ...otherFiles];
          onRequestComposerPrefill(`${intro}\n${all.map((p) => `- ${p}`).join("\n")}`);
          return;
        }
        toasts.pushToast({
          kind: "error",
          title: t(lang, "dropzone.error.noPairs"),
          durationMs: 0,
        });
        if (unpairedAb1.length > 0 || unpairedGb.length > 0) {
          toasts.pushToast({
            kind: "warning",
            title: t(lang, "dropzone.toast.unpaired", {
              ab1: unpairedAb1.length,
              gb: unpairedGb.length,
            }),
          });
        }
        return;
      }

      let response: DroppedFilesResponse | undefined;
      try {
        response = (await window.electronAPI.invoke("analyze-dropped-files", {
          ab1Paths,
          gbPaths,
        })) as DroppedFilesResponse | undefined;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toasts.pushToast({
          kind: "error",
          title: t(lang, "dropzone.toast.error", { message }),
          durationMs: 0,
        });
        return;
      }

      if (response && response.ok) {
        const sampleCount = response.result?.sample_count;
        const count =
          typeof sampleCount === "number"
            ? sampleCount
            : typeof response.pairs === "number"
            ? response.pairs
            : pairs.length;
        toasts.pushToast({
          kind: "success",
          title: t(lang, "dropzone.toast.success", { count }),
        });
      } else {
        const message =
          response && typeof response.error === "string" && response.error.length > 0
            ? response.error
            : "unknown";
        toasts.pushToast({
          kind: "error",
          title: t(lang, "dropzone.toast.error", { message }),
          durationMs: 0,
        });
      }

      if (unpairedAb1.length > 0 || unpairedGb.length > 0) {
        toasts.pushToast({
          kind: "warning",
          title: t(lang, "dropzone.toast.unpaired", {
            ab1: unpairedAb1.length,
            gb: unpairedGb.length,
          }),
        });
      }
    },
    [toasts, onRequestImport],
  );

  useEffect(() => {
    function onDragEnter(event: DragEvent) {
      if (!dragHasFiles(event)) return;
      dragCounterRef.current += 1;
      if (dragCounterRef.current === 1) setDragActive(true);
    }

    function onDragOver(event: DragEvent) {
      if (!dragHasFiles(event)) return;
      // preventDefault is required for the drop event to fire and to keep
      // the browser from navigating to the dropped file.
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    }

    function onDragLeave(event: DragEvent) {
      if (!dragHasFiles(event)) return;
      dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
      if (dragCounterRef.current === 0) setDragActive(false);
    }

    function onDrop(event: DragEvent) {
      if (!dragHasFiles(event)) return;
      event.preventDefault();
      const files = event.dataTransfer?.files;
      resetDrag();
      if (!files || files.length === 0) return;
      void processDrop(files);
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && dragCounterRef.current > 0) {
        resetDrag();
      }
    }

    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [processDrop, resetDrag]);

  return (
    <div className="dropzone-root">
      {children}
      {dragActive ? (
        <div
          className={`dropzone-overlay${reduceMotion ? " dropzone-overlay-static" : ""}`}
          role="status"
          aria-live="polite"
        >
          <div className="dropzone-card">
            <UploadCloud
              size={36}
              strokeWidth={1.6}
              className="dropzone-icon"
              aria-hidden="true"
            />
            <span className="dropzone-hint">{t(language, "dropzone.hint")}</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}
