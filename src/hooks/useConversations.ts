/**
 * Persistent chat-conversation list. Each conversation is a saved snapshot
 * of the assistant↔user message stream the user can revisit, OpenWebUI-style.
 *
 * The agent harness keeps an in-memory `messages` array; this hook mirrors
 * that array into localStorage under a stable conversation id so a refresh
 * (or "New chat" + later switch-back) restores the same dialogue.
 *
 * Storage shape (key: "bioagent.conversations"):
 *   [
 *     { id, title, messages: [{role, content}], createdAt, updatedAt },
 *     ...
 *   ]
 *
 * Storage is best-effort: a corrupt blob falls back to an empty list, and
 * setItem failures are swallowed silently (matches the existing
 * settingsStorage pattern in this codebase).
 */
import { useCallback, useEffect, useRef, useState } from "react";

export interface ConversationMessage {
  role: string;
  content: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ConversationMessage[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "bioagent.conversations";
const CURRENT_KEY = "bioagent.currentConversationId";
const MAX_CONVERSATIONS = 50;

function loadAll(): Conversation[] {
  try {
    const raw = window.localStorage?.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((c): c is Conversation =>
      c && typeof c.id === "string" && Array.isArray(c.messages),
    );
  } catch {
    return [];
  }
}

function saveAll(items: Conversation[]): void {
  try {
    window.localStorage?.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_CONVERSATIONS)));
  } catch {
    // ignore — storage may be full or disabled
  }
}

function loadCurrentId(): string | null {
  try {
    return window.localStorage?.getItem(CURRENT_KEY) || null;
  } catch {
    return null;
  }
}

function saveCurrentId(id: string | null): void {
  try {
    if (id) window.localStorage?.setItem(CURRENT_KEY, id);
    else window.localStorage?.removeItem(CURRENT_KEY);
  } catch {
    // ignore
  }
}

function deriveTitle(messages: ConversationMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return "New chat";
  const text = String(firstUser.content || "").trim().replace(/\s+/g, " ");
  return text.length > 40 ? `${text.slice(0, 40)}…` : (text || "New chat");
}

function genId(): string {
  return `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export interface UseConversationsResult {
  conversations: readonly Conversation[];
  currentId: string | null;
  /** Pick an existing conversation as current; the host should reload the
   *  agent harness with that conversation's messages. */
  setCurrent: (id: string | null) => void;
  /** Persist the latest message snapshot for the current conversation. If
   *  no current conversation exists and messages is non-empty, create one. */
  syncMessages: (messages: ConversationMessage[]) => void;
  /** Start a fresh conversation. Returns the new id. */
  newConversation: () => string;
  /** Drop a conversation by id. If it was current, current resets to null. */
  remove: (id: string) => void;
}

export function useConversations(): UseConversationsResult {
  const [conversations, setConversations] = useState<Conversation[]>(() => loadAll());
  const [currentId, setCurrentIdState] = useState<string | null>(() => loadCurrentId());

  // Persist on every change.
  useEffect(() => {
    saveAll(conversations);
  }, [conversations]);
  useEffect(() => {
    saveCurrentId(currentId);
  }, [currentId]);

  const setCurrent = useCallback((id: string | null) => {
    setCurrentIdState(id);
  }, []);

  // Track latest sync inputs so the debounced effect doesn't fire stale.
  const pendingRef = useRef<{ id: string | null; messages: ConversationMessage[] } | null>(null);

  const syncMessages = useCallback((messages: ConversationMessage[]) => {
    pendingRef.current = { id: currentId, messages };
    setConversations((current) => {
      // Empty → don't create empty conversations.
      if (messages.length === 0) return current;

      const id = currentId || genId();
      const now = Date.now();
      const existing = current.find((c) => c.id === id);
      const title = existing?.title || deriveTitle(messages);
      const updated: Conversation = existing
        ? { ...existing, title: existing.title || deriveTitle(messages), messages, updatedAt: now }
        : { id, title, messages, createdAt: now, updatedAt: now };
      const next = existing
        ? current.map((c) => (c.id === id ? updated : c))
        : [updated, ...current];
      // Sort by updatedAt desc so newest bubbles to top of sidebar.
      next.sort((a, b) => b.updatedAt - a.updatedAt);
      // If the conversation was created on the fly, adopt its id.
      if (!existing && (!currentId || currentId !== id)) {
        // Defer state set to avoid setState-during-render.
        Promise.resolve().then(() => setCurrentIdState(id));
      }
      return next.slice(0, MAX_CONVERSATIONS);
    });
  }, [currentId]);

  const newConversation = useCallback((): string => {
    const id = genId();
    const now = Date.now();
    const fresh: Conversation = {
      id,
      title: "New chat",
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    setConversations((current) => [fresh, ...current].slice(0, MAX_CONVERSATIONS));
    setCurrentIdState(id);
    return id;
  }, []);

  const remove = useCallback((id: string) => {
    setConversations((current) => current.filter((c) => c.id !== id));
    setCurrentIdState((prev) => (prev === id ? null : prev));
  }, []);

  return {
    conversations,
    currentId,
    setCurrent,
    syncMessages,
    newConversation,
    remove,
  };
}
