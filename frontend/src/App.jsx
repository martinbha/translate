import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";

export default function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    api.me().then(setUser).catch(() => {}).finally(() => setChecking(false));
  }, []);

  if (checking) return null;
  if (!user) return <Login onLogin={setUser} />;
  return <Dashboard user={user} onLogout={() => setUser(null)} />;
}

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [totp, setTotp] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setErr("");
    setBusy(true);
    try {
      const u = await api.login(username, password, totp);
      onLogin(u);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wrap">
      <form className="card login" onSubmit={submit}>
        <h1>Sign in</h1>
        <p className="sub">Transcribe &amp; translate audio</p>
        <label>Username</label>
        <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
        <label>Password</label>
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
        <label>Authenticator code</label>
        <input value={totp} onChange={(e) => setTotp(e.target.value)} inputMode="numeric" placeholder="123456" />
        {err && <div className="err">{err}</div>}
        <div style={{ marginTop: 18 }}>
          <button disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        </div>
      </form>
    </div>
  );
}

function Dashboard({ user, onLogout }) {
  const [jobs, setJobs] = useState([]);
  const timer = useRef(null);

  async function refresh() {
    try {
      setJobs(await api.jobs());
    } catch {}
  }

  useEffect(() => {
    refresh();
    timer.current = setInterval(refresh, 2000);
    return () => clearInterval(timer.current);
  }, []);

  async function logout() {
    await api.logout().catch(() => {});
    onLogout();
  }

  return (
    <div className="wrap">
      <div className="topbar">
        <div>
          <h1>Transcribe &amp; Translate</h1>
          <div className="sub">Signed in as {user.username}</div>
        </div>
        <button className="ghost" onClick={logout}>Log out</button>
      </div>

      <Uploader onUploaded={refresh} />

      <div className="card">
        <h1 style={{ fontSize: 17, marginBottom: 6 }}>Jobs</h1>
        {jobs.length === 0 && <div className="sub">No jobs yet. Drop an audio file above.</div>}
        {jobs.map((j) => <JobRow key={j.id} job={j} />)}
      </div>
    </div>
  );
}

function Uploader({ onUploaded }) {
  const [over, setOver] = useState(false);
  const [pct, setPct] = useState(null);
  const [err, setErr] = useState("");
  const inputRef = useRef(null);

  async function send(file) {
    if (!file) return;
    setErr("");
    setPct(0);
    try {
      await api.upload(file, (f) => setPct(Math.round(f * 100)));
      setPct(null);
      onUploaded();
    } catch (e) {
      setErr(e.message);
      setPct(null);
    }
  }

  return (
    <div className="card">
      <div
        className={"drop" + (over ? " over" : "")}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          send(e.dataTransfer.files?.[0]);
        }}
      >
        {pct === null ? (
          <>
            <div style={{ fontSize: 16, marginBottom: 6 }}>Drop an audio file here</div>
            <div className="sub">or click to choose · mp3, wav, m4a, flac, video…</div>
          </>
        ) : (
          <>
            <div style={{ marginBottom: 10 }}>Uploading… {pct}%</div>
            <div className="bar"><div style={{ width: `${pct}%` }} /></div>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="audio/*,video/*"
          style={{ display: "none" }}
          onChange={(e) => send(e.target.files?.[0])}
        />
      </div>
      {err && <div className="err">{err}</div>}
    </div>
  );
}

function JobRow({ job }) {
  const [preview, setPreview] = useState(null);
  const pct = Math.round((job.progress || 0) * 100);

  async function togglePreview() {
    if (preview) return setPreview(null);
    try {
      const r = await api.markdown(job.id);
      setPreview(r.markdown);
    } catch (e) {
      setPreview("Could not load preview: " + e.message);
    }
  }

  return (
    <div className="job">
      <div className="row">
        <div className="fname">{job.original_filename}</div>
        <span className={"pill " + job.status}>{job.status}</span>
      </div>

      {job.status === "processing" || job.status === "queued" ? (
        <>
          <div className="stage">{job.stage} · {pct}%</div>
          <div className="bar"><div style={{ width: `${pct}%` }} /></div>
        </>
      ) : null}

      {job.status === "error" && <div className="err">{job.error}</div>}

      {job.status === "done" && (
        <>
          <div className="stage">
            {job.source_language ? `Source: ${job.source_language} → English` : "English"}
            {job.num_speakers ? ` · ${job.num_speakers} speakers` : ""}
          </div>
          <div className="actions">
            <a href={api.downloadUrl(job.id)}>
              <button>Download .md</button>
            </a>
            <button className="ghost" onClick={togglePreview}>
              {preview ? "Hide" : "Preview"}
            </button>
          </div>
          {preview && <pre className="preview">{preview}</pre>}
        </>
      )}
    </div>
  );
}
