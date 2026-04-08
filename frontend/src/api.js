const BASE = '/api';

export async function analyzeResume(file, anonymize = false) {
  const form = new FormData();
  form.append('file', file);
  form.append('anonymize', anonymize);
  const res = await fetch(`${BASE}/analyze`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).detail || 'Analysis failed');
  return res.json();
}

export async function matchResume(file, { jobDescription, targetRole, location, anonymize }) {
  const form = new FormData();
  form.append('file', file);
  form.append('job_description', jobDescription || '');
  form.append('target_role', targetRole || '');
  form.append('location', location || 'India');
  form.append('anonymize', anonymize || false);
  const res = await fetch(`${BASE}/match`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).detail || 'Match failed');
  return res.json();
}

export async function rankCandidates(files, { jobDescription, targetRole, anonymize }) {
  const form = new FormData();
  files.forEach(f => form.append('files', f));
  form.append('job_description', jobDescription);
  form.append('target_role', targetRole || '');
  form.append('anonymize', anonymize || false);
  const res = await fetch(`${BASE}/rank`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).detail || 'Ranking failed');
  return res.json();
}

export async function getRoles() {
  const res = await fetch(`${BASE}/roles`);
  return res.json();
}

export async function getHistory(limit = 20) {
  const res = await fetch(`${BASE}/history?limit=${limit}`);
  return res.json();
}

export async function clearHistory() {
  const res = await fetch(`${BASE}/history`, { method: 'DELETE' });
  return res.json();
}
