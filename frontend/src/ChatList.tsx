import { useState } from "react";
import type { Api, Conversation, Folder } from "./api";
import { Icon } from "./icons";

type Props = {
  api: Api;
  conversations: Conversation[];
  folders: Folder[];
  activeId: string | null;
  onOpen: (id: string) => void;
  onNewChat: (folderId: string | null) => void;
  onChanged: () => void;
  onDeleted: (id: string) => void;
  onError: (e: unknown) => void;
  onOpenRules: (folderId: string) => void;
};

const DRAG_TYPE = "application/x-amy-chat";

/** Sidebar chat history: collapsible folders (drag chats in and out) plus unfiled chats. */
export function ChatList({ api, conversations, folders, activeId, onOpen, onNewChat, onChanged, onDeleted, onError, onOpenRules }: Props) {
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const [moving, setMoving] = useState<string | null>(null);

  const run = (fn: () => Promise<unknown>) => () =>
    fn()
      .then(onChanged)
      .catch(onError);

  async function newFolder() {
    const name = prompt("New folder name")?.trim();
    if (name) await run(() => api.createFolder(name))();
  }

  async function renameFolder(f: Folder) {
    const name = prompt("Rename folder", f.name)?.trim();
    if (name && name !== f.name) await run(() => api.updateFolder(f.id, { name }))();
  }

  async function deleteFolder(f: Folder) {
    if (confirm(`Delete the folder "${f.name}"? Its chats are kept.`)) await run(() => api.deleteFolder(f.id))();
  }

  async function renameChat(c: Conversation) {
    const title = prompt("Rename chat", c.title)?.trim();
    if (title && title !== c.title) await run(() => api.renameConversation(c.id, title))();
  }

  async function deleteChat(c: Conversation) {
    if (!confirm(`Delete "${c.title}"?`)) return;
    try {
      await api.deleteConversation(c.id);
      onDeleted(c.id);
      onChanged();
    } catch (e) {
      onError(e);
    }
  }

  async function move(chatId: string, folderId: string | null) {
    setMoving(null);
    const c = conversations.find((x) => x.id === chatId);
    if (!c || (c.folder_id ?? null) === folderId) return;
    await run(() => api.moveConversation(chatId, folderId))();
  }

  // Drag and drop: chats are draggable; folders and the unfiled area accept them.
  const dropProps = (target: string | null) => ({
    onDragOver: (e: React.DragEvent) => {
      if (e.dataTransfer.types.includes(DRAG_TYPE)) {
        e.preventDefault();
        setDropTarget(target ?? "unfiled");
      }
    },
    onDragLeave: () => setDropTarget(null),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDropTarget(null);
      const id = e.dataTransfer.getData(DRAG_TYPE);
      if (id) void move(id, target);
    },
  });

  function ChatItem({ c }: { c: Conversation }) {
    return (
      <div
        className={`history-item ${activeId === c.id ? "active" : ""}`}
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData(DRAG_TYPE, c.id);
          e.dataTransfer.effectAllowed = "move";
        }}
      >
        {moving === c.id ? (
          <select
            className="move-select"
            autoFocus
            value={c.folder_id ?? ""}
            onChange={(e) => void move(c.id, e.target.value || null)}
            onBlur={() => setMoving(null)}
          >
            <option value="">No folder</option>
            {folders.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name}
              </option>
            ))}
          </select>
        ) : (
          <button className="history-title" title={c.title} onClick={() => onOpen(c.id)}>
            {c.title}
          </button>
        )}
        <span className="history-actions">
          {folders.length > 0 && (
            <button aria-label="Move to folder" title="Move to folder" onClick={() => setMoving(c.id)}>
              <Icon name="move" size={14} />
            </button>
          )}
          <button aria-label="Rename" title="Rename" onClick={() => void renameChat(c)}>
            <Icon name="edit" size={14} />
          </button>
          <button aria-label="Delete" title="Delete" onClick={() => void deleteChat(c)}>
            <Icon name="close" size={14} />
          </button>
        </span>
      </div>
    );
  }

  const unfiled = conversations.filter((c) => !c.folder_id || !folders.some((f) => f.id === c.folder_id));

  return (
    <div className="history">
      <div className="history-label">
        <span>Chats</span>
        <button className="history-add" title="New folder" onClick={() => void newFolder()}>
          + Folder
        </button>
      </div>

      {folders.map((f) => {
        const chats = conversations.filter((c) => c.folder_id === f.id);
        return (
          <div key={f.id} className={`folder ${dropTarget === f.id ? "drop" : ""}`} {...dropProps(f.id)}>
            <div className="folder-head">
              <button
                className="folder-toggle"
                aria-expanded={!f.collapsed}
                onClick={run(() => api.updateFolder(f.id, { collapsed: !f.collapsed }))}
              >
                <span className="caret">{f.collapsed ? "▸" : "▾"}</span>
                <span className="folder-name">{f.name}</span>
                <span className="folder-count">{chats.length}</span>
              </button>
              <span className="history-actions">
                <button aria-label="New chat in folder" title="New chat in this folder" onClick={() => onNewChat(f.id)}>
                  +
                </button>
                <button aria-label="Folder rules" title="Rules for this folder" onClick={() => onOpenRules(f.id)}>
                  <Icon name="rules" size={14} />
                </button>
                <button aria-label="Rename folder" title="Rename" onClick={() => void renameFolder(f)}>
                  <Icon name="edit" size={14} />
                </button>
                <button aria-label="Delete folder" title="Delete folder (keeps chats)" onClick={() => void deleteFolder(f)}>
                  <Icon name="close" size={14} />
                </button>
              </span>
            </div>
            {!f.collapsed && (
              <div className="folder-chats">
                {chats.map((c) => (
                  <ChatItem key={c.id} c={c} />
                ))}
                {chats.length === 0 && <p className="folder-empty">Drag chats here</p>}
              </div>
            )}
          </div>
        );
      })}

      <div className={`unfiled ${dropTarget === "unfiled" ? "drop" : ""}`} {...dropProps(null)}>
        {folders.length > 0 && unfiled.length > 0 && <div className="history-sublabel">Not in a folder</div>}
        {unfiled.map((c) => (
          <ChatItem key={c.id} c={c} />
        ))}
      </div>
    </div>
  );
}
