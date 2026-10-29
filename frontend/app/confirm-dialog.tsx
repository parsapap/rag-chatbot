"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";

type Request = { title: string; name: string; detail: string; action: string };

// A replacement for window.confirm that matches the app and keeps a Persian name on its own line,
// where the browser box mixed it into English text and scrambled the word order.
export function useConfirm(): [(r: Request) => Promise<boolean>, ReactNode] {
  const ref = useRef<HTMLDialogElement>(null);
  const resolve = useRef<(ok: boolean) => void>(() => {});
  const [request, setRequest] = useState<Request | null>(null);

  // showModal gives focus trapping, Escape and the backdrop for free. It runs once the content is rendered.
  useEffect(() => {
    if (request && !ref.current?.open) ref.current?.showModal();
  }, [request]);

  const ask = (r: Request) =>
    new Promise<boolean>((res) => {
      resolve.current = res;
      setRequest(r);
    });

  const close = (ok: boolean) => {
    ref.current?.close();
    setRequest(null);
    resolve.current(ok);
  };

  const dialog = (
    <dialog
      ref={ref}
      aria-labelledby="confirm-title"
      onCancel={(e) => {
        e.preventDefault();
        close(false);
      }}
      onClick={(e) => e.target === ref.current && close(false)}
      className="m-auto w-[min(26rem,calc(100vw-2rem))] border border-border bg-card p-0 text-foreground backdrop:bg-black/60"
    >
      {request && (
        <div className="flex flex-col gap-4 p-5">
          <h2 id="confirm-title" className="font-medium">
            {request.title}
          </h2>
          <p dir="auto" className="border-s-2 border-input ps-3 font-mono text-[13px] break-all">
            {request.name}
          </p>
          <p className="text-sm text-muted-foreground">{request.detail}</p>
          <div className="flex justify-end gap-3 pt-1">
            {/* autoFocus on Cancel: Enter right after opening does not delete anything. */}
            <button
              type="button"
              autoFocus
              onClick={() => close(false)}
              className="border border-input px-4 py-1.5 text-sm hover:bg-muted"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => close(true)}
              className="bg-destructive px-4 py-1.5 text-sm font-medium text-background hover:opacity-90"
            >
              {request.action}
            </button>
          </div>
        </div>
      )}
    </dialog>
  );

  return [ask, dialog];
}
