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

      <People />

      <div className="card">
        <h1 style={{ fontSize: 17, marginBottom: 6 }}>Jobs</h1>
        {jobs.length === 0 && <div className="sub">No jobs yet. Drop an audio file above.</div>}
        {jobs.map((j) => <JobRow key={j.id} job={j} />)}
      </div>
    </div>
  );
}

function People() {
  const [people, setPeople] = useState([]);
  const [name, setName] = useState("");
  const [open, setOpen] = useState(false);

  async function refresh() {
    try { setPeople(await api.people()); } catch {}
  }
  useEffect(() => { refresh(); }, []);

  async function add(e) {
    e.preventDefault();
    if (!name.trim()) return;
    await api.addPerson(name.trim()).catch(() => {});
    setName("");
    refresh();
  }

  async function enroll(id, file) {
    if (!file) return;
    await api.enroll(id, file).catch((e) => alert("Enroll failed: " + e.message));
    // Voiceprint extraction runs on the worker; refresh shortly after.
    setTimeout(refresh, 4000);
  }

  return (
    <div className="card">
      <div className="row" style={{ display: "flex", justifyContent: "space-between" }}>
        <h1 style={{ fontSize: 17 }}>Known speakers</h1>
        <button className="ghost" onClick={() => setOpen(!open)}>
          {open ? "Hide" : `Manage (${people.length})`}
        </button>
      </div>
      {open && (
        <>
          <div className="sub" style={{ margin: "6px 0 12px" }}>
            Enroll a clean voice sample per person, or just name speakers on a
            transcript below — either way their voice is learned and auto-labeled next time.
          </div>
          {people.map((p) => (
            <div className="job" key={p.id}>
              <div className="row" style={{ display: "flex", justifyContent: "space-between" }}>
                <div className="fname">{p.name}</div>
                <div className="actions">
                  <span className="sub">{p.voiceprints} voiceprint{p.voiceprints === 1 ? "" : "s"}</span>
                  <label className="ghost" style={{ padding: "6px 10px", borderRadius: 8, cursor: "pointer", border: "1px solid var(--border)" }}>
                    Enroll sample
                    <input type="file" accept="audio/*,video/*" style={{ display: "none" }}
                      onChange={(e) => enroll(p.id, e.target.files?.[0])} />
                  </label>
                  <button className="ghost" onClick={async () => { await api.deletePerson(p.id).catch(()=>{}); refresh(); }}>Delete</button>
                </div>
              </div>
            </div>
          ))}
          <form onSubmit={add} style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <input placeholder="Add a person (name)" value={name} onChange={(e) => setName(e.target.value)} />
            <button>Add</button>
          </form>
        </>
      )}
    </div>
  );
}

function Speakers({ jobId, onRenamed }) {
  const [speakers, setSpeakers] = useState([]);
  const [people, setPeople] = useState([]);

  async function refresh() {
    try {
      const [s, p] = await Promise.all([api.speakers(jobId), api.people()]);
      setSpeakers(s);
      setPeople(p);
    } catch {}
  }
  useEffect(() => { refresh(); }, [jobId]);

  async function assign(sid, value) {
    if (!value) return;
    let body;
    if (value === "new:") {
      const name = prompt("New person's name:");
      if (!name || !name.trim()) return;
      body = { new_name: name.trim() };
    } else {
      body = { person_id: Number(value) };
    }
    await api.assignSpeaker(jobId, sid, body).catch((e) => alert(e.message));
    await refresh();
    onRenamed && onRenamed();
  }

  if (!speakers.length) return null;
  return (
    <div style={{ marginTop: 10 }}>
      <div className="sub" style={{ marginBottom: 6 }}>Speakers — listen and assign names:</div>
      {speakers.map((s) => (
        <div key={s.id} className="row" style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span style={{ minWidth: 90, fontWeight: 600 }}>{s.display_name}</span>
          {s.auto_matched && <span className="pill done">auto {Math.round((s.match_score || 0) * 100)}%</span>}
          <audio controls preload="none" src={api.sampleUrl(jobId, s.id)} style={{ height: 30, maxWidth: 200 }} />
          <select defaultValue="" onChange={(e) => assign(s.id, e.target.value)}>
            <option value="" disabled>Assign…</option>
            {people.map((p) => <option key={p.id} value={String(p.id)}>{p.name}</option>)}
            <option value="new:">+ New person…</option>
          </select>
        </div>
      ))}
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

  async function loadPreview() {
    try {
      const r = await api.markdown(job.id);
      setPreview(r.markdown);
    } catch (e) {
      setPreview("Could not load preview: " + e.message);
    }
  }

  async function togglePreview() {
    if (preview) return setPreview(null);
    loadPreview();
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
          <Speakers jobId={job.id} onRenamed={() => preview && loadPreview()} />
          {preview && <pre className="preview">{preview}</pre>}
        </>
      )}
    </div>
  );
}
