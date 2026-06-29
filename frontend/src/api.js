// Thin fetch wrapper. Cookies are httpOnly, so we just rely on the browser
// sending them with credentials: "include".

async function req(path, opts = {}) {
  const res = await fetch(path, { credentials: "include", ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

export const api = {
  me: () => req("/api/me"),
  login: (username, password, totp) =>
    req("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, totp }),
    }),
  logout: () => req("/api/logout", { method: "POST" }),
  jobs: () => req("/api/jobs"),
  job: (id) => req(`/api/jobs/${id}`),
  markdown: (id) => req(`/api/jobs/${id}/markdown`),
  upload: (file, onProgress) =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/jobs");
      xhr.withCredentials = true;
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress)
          onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
        else {
          let msg = xhr.statusText;
          try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
          reject(new Error(msg));
        }
      };
      xhr.onerror = () => reject(new Error("Upload failed"));
      xhr.send(form);
    }),
  downloadUrl: (id) => `/api/jobs/${id}/download`,

  // People (known speakers)
  people: () => req("/api/people"),
  addPerson: (name) =>
    req("/api/people", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  deletePerson: (id) => req(`/api/people/${id}`, { method: "DELETE" }),
  enroll: (id, file) => {
    const f = new FormData();
    f.append("file", file);
    return req(`/api/people/${id}/enroll`, { method: "POST", body: f });
  },

  // Per-job speakers
  speakers: (jobId) => req(`/api/jobs/${jobId}/speakers`),
  assignSpeaker: (jobId, speakerId, body) =>
    req(`/api/jobs/${jobId}/speakers/${speakerId}/assign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  sampleUrl: (jobId, speakerId) =>
    `/api/jobs/${jobId}/speakers/${speakerId}/sample`,
};
