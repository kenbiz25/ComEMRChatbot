
import { useState, useEffect, useRef } from "react";
import { SunIcon, MoonIcon, ArrowUpIcon, Bars3Icon, XMarkIcon } from "@heroicons/react/24/solid"; // npm i @heroicons/react

export default function App() {
  // ---------------- Theme ----------------
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "light");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  // ---------------- Sidebar ----------------
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // ---------------- Sessions ----------------
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const chatRef = useRef(null);

  useEffect(() => loadSessions(), []);
  useEffect(() => {
    chatRef.current?.scrollTo(0, chatRef.current.scrollHeight);
  }, [messages]);

  async function loadSessions() {
    try {
      const res = await fetch("/sessions");
      const data = await res.json();

      // Newest first by created (backend returns "created")
      const sorted = [...data].sort((a, b) => new Date(b.created || 0) - new Date(a.created || 0));

      setSessions(sorted);

      // If nothing open yet, open the latest session
      if (!currentSessionId && sorted.length > 0) openSession(sorted[0].id);
    } catch (e) {
      console.error("Failed to load sessions", e);
    }
  }

  async function createNewSession() {
    try {
      // Send a name so it looks good in the sidebar
      const res = await fetch("/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "ComEMR Copilot" }),
      });
      const data = await res.json();
      setCurrentSessionId(data.id);
      setMessages([]);
      await loadSessions();
      setSidebarOpen(false);
    } catch (e) {
      console.error("Failed to create session", e);
    }
  }

  async function openSession(id) {
    try {
      // Get session meta
      const res = await fetch(`/sessions/${id}`);
      const sess = await res.json();
      setCurrentSessionId(sess.id);

      // ✅ Load stored messages for this session
      const msgsRes = await fetch(`/sessions/${id}/messages`);
      const msgs = await msgsRes.json();
      setMessages(msgs);

      setSidebarOpen(false);
    } catch (e) {
      console.error("Failed to open session", e);
    }
  }

  // ---------------- Chat ----------------
  const [prompt, setPrompt] = useState("");
  const [role, setRole] = useState("chw");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function sendPrompt() {
    const p = prompt.trim();
    if (!p || loading) return;
    if (!currentSessionId) await createNewSession();

    setLoading(true);
    setError("");
    setMessages(prev => [...prev, { role: "user", content: p }]);
    setPrompt("");

    try {
      const form = new FormData();
      form.append("prompt", p);
      form.append("role", role);
      if (currentSessionId) form.append("session_id", currentSessionId);

      const res = await fetch("/chat", { method: "POST", body: form });
      if (!res.body) throw new Error("Streaming not supported");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullText = "";
      setMessages(prev => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        fullText += decoder.decode(value, { stream: true });
        setMessages(prev => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: fullText };
          return copy;
        });
      }

      fullText += decoder.decode();
      setMessages(prev => {
        const copy = [...prev];
        copy[copy.length - 1] = { role: "assistant", content: fullText };
        return copy;
      });

      await loadSessions();
    } catch (e) {
      setError(`Network error: ${String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  }

  // ---------------- UI ----------------
  return (
    <div className="app-container flex h-screen text-zinc-100 bg-zinc-950">
      {/* Sidebar */}
      <aside className={`sidebar fixed md:relative z-50 top-0 left-0 h-full w-64 bg-zinc-900 p-4 flex flex-col transform transition-transform duration-300 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0`}>
        <div className="flex justify-between items-center mb-4">
          <div className="text-lg font-bold">V.1</div>
          <button className="md:hidden" onClick={() => setSidebarOpen(false)}>
            <XMarkIcon className="w-5 h-5"/>
          </button>
        </div>

        <button onClick={createNewSession} className="mb-4 p-2 rounded bg-blue-600 hover:bg-blue-500 text-sm">
          + New Chat
        </button>

        {/* ✅ Full sessions list for the sidebar */}
        <div className="flex-1 overflow-auto mb-4">
          {sessions.length === 0 ? (
            <div className="text-zinc-500 text-sm">No chats yet</div>
          ) : (
            sessions.map(s => (
              <button
                key={s.id}
                onClick={() => openSession(s.id)}
                className={`w-full text-left p-2 rounded text-sm mb-1 ${s.id === currentSessionId ? "bg-zinc-800" : "hover:bg-zinc-700"}`}
                title={new Date(s.created).toLocaleString()}
              >
                {/* Show stored session name from backend */}
                {s.name || "Untitled"}
              </button>
            ))
          )}
        </div>

        <div className="flex items-center gap-2 text-xs">
          <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
            {theme === "light" ? <MoonIcon className="w-4 h-4"/> : <SunIcon className="w-4 h-4"/>}
          </button>
          Theme: {theme}
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && <div className="fixed inset-0 bg-black opacity-30 md:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* Chat Area */}
      <div className="flex-1 flex flex-col md:ml-64">
        <header className="border-b border-zinc-800 p-3 flex justify-between items-center sticky top-0 bg-zinc-950 z-10">
          <div className="flex items-center gap-2">
            <button className="md:hidden" onClick={() => setSidebarOpen(true)}>
              <Bars3Icon className="w-5 h-5"/>
            </button>

            {/* ✅ ComEMR logo from Vite public/ */}
            <img src="/comemr.png" alt="ComEMR Logo" className="w-8 h-8 mr-2"/>

            <span className="font-medium text-lg">ComEMR Support</span>
          </div>
          <select value={role} onChange={e => setRole(e.target.value)} className="bg-zinc-900 text-sm p-1 rounded">
            <option value="chw">CHW/Peer Supervisor</option>
            <option value="clinician">Clinician</option>
            <option value="gov">Ministry Officials</option>
            {/* if you want to expose clinician explicitly */}
          </select>
        </header>

        <main ref={chatRef} className="flex-1 overflow-auto p-4 flex flex-col gap-4">
          {messages.length === 0 ? (
            <div className="text-zinc-500 text-sm">Start a conversation below.</div>
          ) : (
            messages.map((m, i) => (
              <div key={i} className={`max-w-3xl ${m.role === "user" ? "ml-auto text-right" : ""}`}>
                <div className={`inline-block p-3 rounded-xl text-sm whitespace-pre-wrap ${m.role === "user" ? "bg-blue-600" : "bg-zinc-800"}`}>
                  {m.content}
                </div>
              </div>
            ))
          )}
          {loading && <div className="text-zinc-500 text-sm">Ask ComEMR…</div>}
        </main>

        <footer className="border-t border-zinc-800 p-3 flex gap-2 items-center sticky bottom-0 bg-zinc-950 z-10">
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={onKeyDown}
            rows={2}
            className="flex-1 rounded bg-zinc-900 p-2 text-sm outline-none"
            placeholder="Ask ComEMR..."
          />
          <button
            onClick={sendPrompt}
            disabled={loading || !prompt.trim()}
            className="rounded bg-blue-600 px-4 text-sm hover:bg-blue-500"
          >
            <ArrowUpIcon className="w-4 h-4"/>
          </button>
        </footer>

        {error && <div className="bg-red-600 text-white p-2 text-sm">{error}</div>}
           </div>
    </div>
  );
}