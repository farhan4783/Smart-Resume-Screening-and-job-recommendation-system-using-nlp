import { useState, useEffect, useRef, useCallback } from 'react';
import { analyzeResume, matchResume, rankCandidates, getRoles, getHistory, clearHistory } from './api';
import './index.css';

/* ════════ Small reusable components ════════ */

function Pipeline({ step = -1 }) {
  const steps = ['📄 Parse', '🔍 Extract', '🏷️ Classify', '🧮 Embed', '📊 Match', '💡 Recommend'];
  return (
    <div className="pipeline">
      {steps.map((s, i) => (
        <span key={i}>
          <span className={`pipeline-step${i <= step ? ' active' : ''}`}>{s}</span>
          {i < steps.length - 1 && <span className="pipeline-arrow"> → </span>}
        </span>
      ))}
    </div>
  );
}

function SkillTags({ skills = [], detailed = false, missing = false }) {
  if (!skills.length) return null;
  return (
    <div className="skills-container">
      {skills.map((s, i) => {
        if (missing) return <span key={i} className="skill-missing">✗ {typeof s === 'object' ? s.name : s}</span>;
        if (detailed && typeof s === 'object') {
          const dotClass = `dot-${(s.level || 'familiar').toLowerCase()}`;
          return <span key={i} className="skill-tag"><span className={`skill-dot ${dotClass}`}></span>{s.name}</span>;
        }
        return <span key={i} className="skill-tag">{typeof s === 'object' ? s.name : s}</span>;
      })}
    </div>
  );
}

function ATSRing({ score = 0, grade = 'F' }) {
  const color = score >= 70 ? 'var(--green)' : score >= 50 ? 'var(--amber)' : 'var(--red)';
  const bg = score >= 70 ? 'rgba(34,197,94,0.1)' : score >= 50 ? 'rgba(245,158,11,0.1)' : 'rgba(239,68,68,0.1)';
  return (
    <div>
      <div className="ats-ring" style={{ border: `3px solid ${color}`, background: bg }}>
        <div className="ats-score-num" style={{ color }}>{score}</div>
        <div className="ats-grade" style={{ color }}>Grade {grade}</div>
      </div>
      <div className="ats-label">ATS Score</div>
    </div>
  );
}

function ScoreBreakdown({ breakdown = {} }) {
  const colors = {
    'Contact Info': '#3b82f6', 'Skills Section': '#8b5cf6', 'Experience Detail': '#6366f1',
    'Education': '#a78bfa', 'Format Quality': '#06b6d4', 'Keyword Match': '#f59e0b', 'Online Presence': '#22c55e'
  };
  return (
    <div>
      {Object.entries(breakdown).map(([name, data]) => {
        const pct = data.max > 0 ? (data.score / data.max) * 100 : 0;
        return (
          <div className="score-bar" key={name}>
            <div className="score-bar-header"><span>{name}</span><span>{data.score}/{data.max}</span></div>
            <div className="score-bar-track">
              <div className="score-bar-fill" style={{ width: `${pct}%`, background: colors[name] || '#6366f1' }}></div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Spinner({ text = 'Loading...' }) {
  return (
    <div className="spinner">
      <div className="spinner-dot"></div>
      <div className="spinner-dot"></div>
      <div className="spinner-dot"></div>
      <span>{text}</span>
    </div>
  );
}

function FileUpload({ onFile, accept = '.pdf,.docx,.doc', multiple = false, label }) {
  const ref = useRef();
  const [fileName, setFileName] = useState('');

  const handleChange = (e) => {
    const files = e.target.files;
    if (multiple) {
      setFileName(`${files.length} file(s) selected`);
      onFile(Array.from(files));
    } else if (files[0]) {
      setFileName(files[0].name);
      onFile(files[0]);
    }
  };

  return (
    <div className="file-upload" onClick={() => ref.current.click()}>
      <input ref={ref} type="file" accept={accept} multiple={multiple} onChange={handleChange} />
      <div className="file-upload-icon">📄</div>
      <div className="file-upload-text">{fileName || (label || 'Click to upload resume')}</div>
      <div className="file-upload-sub">Supports PDF and DOCX</div>
    </div>
  );
}

function scoreColor(score) {
  if (score >= 0.70) return 'var(--green)';
  if (score >= 0.50) return 'var(--amber)';
  return 'var(--red)';
}

/* ════════ Job Seeker Mode ════════ */

function SeekerDashboard({ anonymize }) {
  const [file, setFile] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [matchData, setMatchData] = useState(null);
  const [roles, setRoles] = useState([]);
  const [selectedRole, setSelectedRole] = useState('');
  const [location, setLocation] = useState('India');
  const [tab, setTab] = useState('profile');
  const [loading, setLoading] = useState('');

  useEffect(() => { getRoles().then(r => setRoles(r.roles || [])); }, []);

  const handleUpload = useCallback(async (f) => {
    setFile(f);
    setMatchData(null);
    setLoading('Analyzing resume...');
    try {
      const data = await analyzeResume(f, anonymize);
      setProfileData(data);
      if (data.category?.category && roles.length) {
        const cat = data.category.category;
        setSelectedRole(roles.includes(cat) ? cat : roles[0]);
      }
    } catch (e) {
      alert('Error: ' + e.message);
    }
    setLoading('');
  }, [anonymize, roles]);

  const handleMatch = async () => {
    if (!file) return;
    setLoading('Running match analysis...');
    try {
      const data = await matchResume(file, { targetRole: selectedRole, location, anonymize });
      setMatchData(data);
      setTab('match');
    } catch (e) {
      alert('Error: ' + e.message);
    }
    setLoading('');
  };

  if (loading) return <><Pipeline step={loading.includes('match') ? 4 : 2} /><Spinner text={loading} /></>;

  if (!profileData) {
    return (
      <div>
        <h3 style={{ marginBottom: '1rem' }}>🎯 Job Seeker Dashboard</h3>
        <FileUpload onFile={handleUpload} />
        <div className="empty-state mt-3">
          <div className="empty-state-icon">📄</div>
          <div className="empty-state-title">Upload your resume to get started</div>
          <div className="empty-state-sub">
            Our NLP pipeline will extract your profile, predict your role category,
            compute an ATS score, and find matching jobs — all in seconds.
          </div>
          <Pipeline step={-1} />
        </div>
      </div>
    );
  }

  const { entities, category, ats, validation } = profileData;
  const tabs = [
    { id: 'profile', label: '📊 Profile & ATS' },
    { id: 'match', label: '🎯 Match Analysis' },
    { id: 'upskill', label: '📚 Upskill Path' },
    { id: 'jobs', label: '💼 Live Jobs' },
  ];

  return (
    <div>
      <Pipeline step={matchData ? 5 : 3} />
      {validation?.is_valid
        ? <div className="alert-success mb-2">✅ Valid Resume — Confidence: {validation.score}/100</div>
        : <div className="alert-warning mb-2">⚠️ {validation?.reason}</div>}

      <div className="tabs">
        {tabs.map(t => (
          <button key={t.id} className={`tab${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {/* ── TAB: Profile ── */}
      {tab === 'profile' && (
        <div className="row">
          <div className="col-2">
            <div className="profile-card">
              <div className="profile-name">{entities?.name || 'Candidate'}</div>
              <div className="profile-role">
                {entities?.job_titles?.slice(0, 2).join(', ') || category?.category?.replace(/-/g, ' ')} · {entities?.experience_years || 0}+ years
              </div>
              <div>
                <span className="category-badge">{category?.category || 'N/A'}</span>
                <span className="category-badge-outline">Confidence: {Math.round((category?.confidence || 0) * 100)}%</span>
              </div>
            </div>

            <div className="metrics-row mt-2">
              <div className="metric-card"><div className="metric-value text-sm">📧 {entities?.email || 'N/A'}</div><div className="metric-label">Email</div></div>
              <div className="metric-card"><div className="metric-value text-sm">📱 {entities?.phone || 'N/A'}</div><div className="metric-label">Phone</div></div>
              <div className="metric-card"><div className="metric-value text-sm">🎓 {entities?.education?.slice(0, 2).join(', ') || 'N/A'}</div><div className="metric-label">Education</div></div>
            </div>

            {entities?.links?.linkedin && <a href={entities.links.linkedin} className="job-apply" target="_blank" rel="noreferrer">🔗 LinkedIn</a>}
            {entities?.links?.github && <a href={entities.links.github} className="job-apply" target="_blank" rel="noreferrer" style={{ marginLeft: 12 }}>💻 GitHub</a>}

            {category?.top_3?.length > 0 && (
              <div className="mt-2">
                <div className="section-title">🏷️ Category Prediction (Top 3)</div>
                {category.top_3.map(([name, prob]) => (
                  <div className="score-bar" key={name}>
                    <div className="score-bar-header"><span>{name}</span><span>{Math.round(prob * 100)}%</span></div>
                    <div className="score-bar-track">
                      <div className="score-bar-fill" style={{ width: `${prob * 100}%`, background: prob >= 0.3 ? '#6366f1' : '#475569' }}></div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="section-title mt-2">🛠️ Skills Detected</div>
            <SkillTags skills={entities?.skills_detailed || []} detailed={true} />
            {entities?.skills_detailed?.length > 0 && (
              <div className="text-xs text-muted mt-1">
                <span className="skill-dot dot-expert" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%' }}></span> Expert &nbsp;
                <span className="skill-dot dot-proficient" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%' }}></span> Proficient &nbsp;
                <span className="skill-dot dot-intermediate" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%' }}></span> Intermediate &nbsp;
                <span className="skill-dot dot-familiar" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%' }}></span> Familiar
              </div>
            )}

            {entities?.organizations?.length > 0 && (
              <>
                <div className="section-title mt-2">🏢 Organizations</div>
                <SkillTags skills={entities.organizations} />
              </>
            )}
          </div>

          <div className="col">
            <ATSRing score={ats?.total_score || 0} grade={ats?.grade || 'F'} />
            <div className="mt-2"><ScoreBreakdown breakdown={ats?.breakdown || {}} /></div>
            {ats?.suggestions?.length > 0 && (
              <div className="mt-2">
                <div className="section-title" style={{ fontSize: '0.9rem' }}>💡 Quick Fixes</div>
                {ats.suggestions.slice(0, 3).map((tip, i) => (
                  <div key={i} className="text-xs text-muted" style={{ padding: '2px 0' }}>• {tip}</div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── TAB: Match ── */}
      {tab === 'match' && (
        <div>
          <div className="row mb-2">
            <div className="col-2">
              <label className="text-sm text-muted">🎯 Target Role</label>
              <select className="select-input mt-1" value={selectedRole} onChange={e => setSelectedRole(e.target.value)}>
                {roles.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div className="col">
              <label className="text-sm text-muted">📍 Location</label>
              <input className="text-input mt-1" value={location} onChange={e => setLocation(e.target.value)} />
            </div>
          </div>
          <button className="btn-primary mb-2" onClick={handleMatch} disabled={!file}>🚀 Analyze Match & Find Jobs</button>

          {matchData && (
            <div>
              {/* Gap Analysis */}
              <div className="row mt-2">
                <div className="col">
                  <div className="metrics-row">
                    <div className="metric-card"><div className="metric-value text-green">{matchData.gap?.matched_skills?.length || 0}</div><div className="metric-label">Skills Matched</div></div>
                    <div className="metric-card"><div className="metric-value text-red">{matchData.gap?.missing_skills?.length || 0}</div><div className="metric-label">Skills Missing</div></div>
                    <div className="metric-card"><div className="metric-value text-accent">{matchData.gap?.coverage_pct || 0}%</div><div className="metric-label">Coverage</div></div>
                  </div>
                </div>
                <div className="col-2">
                  {matchData.gap?.matched_skills?.length > 0 && <><div className="fw-700 text-sm text-green mb-1">✅ You Have:</div><SkillTags skills={matchData.gap.matched_skills} /></>}
                  {matchData.gap?.missing_skills?.length > 0 && <><div className="fw-700 text-sm text-red mt-1 mb-1">❌ Missing:</div><SkillTags skills={matchData.gap.missing_skills} missing /></>}
                </div>
              </div>

              {/* Hybrid Score */}
              {matchData.jobs?.[0] && (
                <>
                  <div className="section-title">📊 Hybrid Score Breakdown (Top Match)</div>
                  <div className="metrics-row">
                    <div className="metric-card"><div className="metric-value" style={{ color: scoreColor(matchData.jobs[0].score) }}>{Math.round(matchData.jobs[0].score * 100)}%</div><div className="metric-label">Total Score</div></div>
                    <div className="metric-card"><div className="metric-value text-accent">{Math.round((matchData.jobs[0].semantic_score || 0) * 100)}%</div><div className="metric-label">S-BERT Cosine (60%)</div></div>
                    <div className="metric-card"><div className="metric-value" style={{ color: '#a78bfa' }}>{Math.round((matchData.jobs[0].skill_score || 0) * 100)}%</div><div className="metric-label">Skill Overlap (25%)</div></div>
                    <div className="metric-card"><div className="metric-value" style={{ color: '#c084fc' }}>{Math.round((matchData.jobs[0].category_score || 0) * 100)}%</div><div className="metric-label">Category (15%)</div></div>
                  </div>
                </>
              )}

              {/* AI Explanation */}
              {matchData.explanation && (
                <>
                  <div className="section-title">🤖 AI Match Explanation</div>
                  <div className="glass-card" dangerouslySetInnerHTML={{ __html: matchData.explanation.replace(/\n/g, '<br/>') }}></div>
                </>
              )}

              {/* Resume Feedback */}
              {matchData.resume_feedback && (
                <>
                  <div className="section-title">💡 AI Resume Tips</div>
                  <div className="glass-card" dangerouslySetInnerHTML={{ __html: matchData.resume_feedback.replace(/\n/g, '<br/>') }}></div>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── TAB: Upskill ── */}
      {tab === 'upskill' && (
        <div>
          {matchData?.courses?.length > 0 ? (
            <>
              <div className="section-title">📚 Recommended Learning Path</div>
              <div className="text-sm text-muted mb-2">Based on {matchData.gap?.missing_skills?.length || 0} skill gaps identified.</div>
              {matchData.courses.map((item, i) => (
                <Expandable key={i} title={`📖 ${item.skill}`}>
                  {item.courses.map((c, j) => (
                    <div key={j} style={{ padding: '4px 0' }}>
                      🎓 <strong>{c.platform}</strong> — <a href={c.url} target="_blank" rel="noreferrer" className="job-apply">{c.title}</a>
                    </div>
                  ))}
                </Expandable>
              ))}
            </>
          ) : matchData ? (
            <div className="alert-success">🎉 No skill gaps identified — you're fully covered!</div>
          ) : (
            <div className="alert-info">👆 Run the Match Analysis first to see your personalized upskill path.</div>
          )}
        </div>
      )}

      {/* ── TAB: Jobs ── */}
      {tab === 'jobs' && (
        <div>
          {matchData?.jobs?.length > 0 ? (
            <>
              <div className="section-title">💼 Live Job Matches</div>
              {matchData.jobs.slice(0, 6).map((job, i) => (
                <div className="job-card" key={i}>
                  <div>
                    <div className="job-title">{job.job_title}</div>
                    <div className="job-meta">🏢 {job.employer_name} · 📍 {job.job_city} · {(job.job_employment_type || '').replace(/_/g, ' ')}</div>
                    <div className="job-date">📅 {job.job_posted_at?.split('T')[0] || job.job_posted_at}</div>
                    <a href={job.job_apply_link} target="_blank" rel="noreferrer" className="job-apply">🔗 Apply Now</a>
                  </div>
                  <div>
                    <div className="job-score" style={{ color: job.color }}>{Math.round(job.score * 100)}%</div>
                    <div className="job-score-label" style={{ color: job.color }}>{job.label}</div>
                  </div>
                </div>
              ))}
            </>
          ) : (
            <div className="alert-info">👆 Run the Match Analysis to discover live job opportunities.</div>
          )}
        </div>
      )}
    </div>
  );
}

/* ════════ Expandable ════════ */

function Expandable({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="expandable">
      <div className="expandable-header" onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span className={`expandable-chevron${open ? ' open' : ''}`}>▼</span>
      </div>
      {open && <div className="expandable-body">{children}</div>}
    </div>
  );
}

/* ════════ Recruiter Mode ════════ */

function RecruiterDashboard({ anonymize }) {
  const [files, setFiles] = useState([]);
  const [jd, setJd] = useState('');
  const [roles, setRoles] = useState([]);
  const [targetRole, setTargetRole] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState('');

  useEffect(() => { getRoles().then(r => { setRoles(r.roles || []); if (r.roles?.length) setTargetRole(r.roles[0]); }); }, []);

  const handleRank = async () => {
    if (!files.length || !jd) return;
    setLoading('Ranking candidates...');
    try {
      const data = await rankCandidates(files, { jobDescription: jd, targetRole, anonymize });
      setResult(data);
    } catch (e) {
      alert('Error: ' + e.message);
    }
    setLoading('');
  };

  if (loading) return <><Pipeline step={4} /><Spinner text={loading} /></>;

  return (
    <div>
      <h3 style={{ marginBottom: '0.5rem' }}>🏢 Recruiter Dashboard</h3>
      <p className="text-sm text-muted mb-2">Paste a job description, upload candidate resumes, and rank them by hybrid semantic fit.</p>

      <div className="row mb-2">
        <div className="col">
          <label className="text-sm text-muted mb-1" style={{ display: 'block' }}>📋 Job Description</label>
          <textarea className="text-area" placeholder="e.g. We are looking for a Senior Python Developer with 5+ years in Django, REST APIs, Docker, and AWS..." value={jd} onChange={e => setJd(e.target.value)} />
        </div>
        <div className="col">
          <label className="text-sm text-muted mb-1" style={{ display: 'block' }}>📂 Candidate Resumes</label>
          <FileUpload onFile={setFiles} multiple label="Click to upload resumes (multiple)" />
        </div>
      </div>

      <div className="row mb-2 gap-8">
        <div className="col">
          <label className="text-sm text-muted">🎯 Role Category</label>
          <select className="select-input mt-1" value={targetRole} onChange={e => setTargetRole(e.target.value)}>
            {roles.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
        <div className="col">
          <button className="btn-primary mt-2" onClick={handleRank} disabled={!jd || !files.length}>🏃 Rank Candidates</button>
        </div>
      </div>

      {result && (
        <div>
          <Pipeline step={5} />
          <hr className="divider" />

          <div className="section-title">🏆 Candidate Ranking</div>
          <div className="metrics-row">
            <div className="metric-card"><div className="metric-value">{result.total}</div><div className="metric-label">Total</div></div>
            <div className="metric-card"><div className="metric-value text-green">{result.shortlisted}</div><div className="metric-label">Shortlisted (≥70%)</div></div>
            <div className="metric-card"><div className="metric-value text-accent">{Math.round(result.top_score * 100)}%</div><div className="metric-label">Top Score</div></div>
            <div className="metric-card"><div className="metric-value">{Math.round(result.avg_score * 100)}%</div><div className="metric-label">Avg Score</div></div>
          </div>

          {/* Table */}
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th><th>Candidate</th><th>Category</th><th>Match</th><th>ATS</th><th>Exp</th><th>Skills</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {result.candidates.map((c, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td style={{ fontWeight: 600 }}>{c.name}</td>
                  <td><span className="category-badge-outline">{c.predicted_category}</span></td>
                  <td style={{ color: c.color, fontWeight: 700 }}>{Math.round(c.match_score * 100)}%</td>
                  <td>{c.ats_score} ({c.ats_grade})</td>
                  <td>{c.experience_years}y</td>
                  <td>{c.skills?.length || 0}</td>
                  <td>{c.shortlisted ? '✅' : '❌'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Bar Chart */}
          <div className="section-title">📊 Score Comparison</div>
          <div className="bar-chart" style={{ position: 'relative' }}>
            {result.candidates.map((c, i) => (
              <div className="bar-chart-col" key={i}>
                <div className="bar-chart-value" style={{ color: c.color }}>{Math.round(c.match_score * 100)}%</div>
                <div className="bar-chart-bar" style={{
                  height: `${c.match_score * 100 * 1.7}px`,
                  background: c.color,
                }}></div>
                <div className="bar-chart-label">{c.name?.slice(0, 12)}</div>
              </div>
            ))}
          </div>

          {/* Deep Dives */}
          <div className="section-title">🔍 Candidate Deep Dives</div>
          {result.candidates.map((c, i) => (
            <Expandable key={i} title={`${c.shortlisted ? '✅' : '❌'} ${c.name} — ${Math.round(c.match_score * 100)}% | ATS: ${c.ats_score} | ${c.predicted_category}`} defaultOpen={i === 0}>
              <div className="row">
                <div className="col">
                  <ATSRing score={c.ats_score} grade={c.ats_grade} />
                </div>
                <div className="col-2">
                  <p className="text-sm">📧 <strong>Email:</strong> {c.email || 'N/A'}</p>
                  <p className="text-sm">🎓 <strong>Education:</strong> {c.education?.join(', ') || 'N/A'}</p>
                  <p className="text-sm">🏷️ <strong>Category:</strong> {c.predicted_category}</p>
                  <p className="text-sm">⏱️ <strong>Experience:</strong> {c.experience_years} years</p>
                  {c.skills?.length > 0 && <><div className="fw-700 text-sm mt-1">🛠️ Skills:</div><SkillTags skills={c.skills} /></>}
                </div>
              </div>

              <div className="row mt-2">
                <div className="col">
                  {c.gap?.matched_skills?.length > 0 && <><div className="fw-700 text-sm text-green mb-1">✅ Matched:</div><SkillTags skills={c.gap.matched_skills} /></>}
                </div>
                <div className="col">
                  {c.gap?.missing_skills?.length > 0 && <><div className="fw-700 text-sm text-red mb-1">❌ Missing:</div><SkillTags skills={c.gap.missing_skills} missing /></>}
                </div>
              </div>

              {c.ai_explanation && (
                <div className="mt-2">
                  <div className="fw-700 text-sm mb-1">🤖 AI Explanation:</div>
                  <div className="glass-card" dangerouslySetInnerHTML={{ __html: c.ai_explanation.replace(/\n/g, '<br/>') }}></div>
                </div>
              )}
            </Expandable>
          ))}
        </div>
      )}
    </div>
  );
}

/* ════════ Main App ════════ */

export default function App() {
  const [mode, setMode] = useState('seeker');
  const [anonymize, setAnonymize] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => { getHistory(8).then(h => setHistory(h.history || [])); }, []);

  const handleClearHistory = async () => {
    await clearHistory();
    setHistory([]);
  };

  return (
    <div className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">🧠</div>
          <div className="sidebar-brand-name">ResumeIQ</div>
          <div className="sidebar-brand-sub">Intelligent Resume Screening v2.0</div>
        </div>

        <hr className="sidebar-divider" />
        <div className="sidebar-label">Mode</div>
        <button className={`mode-btn${mode === 'seeker' ? ' active' : ''}`} onClick={() => setMode('seeker')}>🎯 Job Seeker</button>
        <button className={`mode-btn${mode === 'recruiter' ? ' active' : ''}`} onClick={() => setMode('recruiter')}>🏢 Recruiter</button>

        <hr className="sidebar-divider" />
        <div className="sidebar-label">Settings</div>
        <div className="toggle-row">
          <span className="toggle-label">🙈 Blind Screening</span>
          <div className={`toggle-switch${anonymize ? ' on' : ''}`} onClick={() => setAnonymize(!anonymize)}></div>
        </div>

        <hr className="sidebar-divider" />
        <div className="sidebar-label">Recent Sessions</div>
        {history.length > 0 ? (
          <>
            {history.slice(0, 5).map((h, i) => (
              <div key={i} className="history-item">📄 {h.filename?.slice(0, 18)}… · {h.match_score ? Math.round(h.match_score * 100) + '%' : '—'}</div>
            ))}
            <button className="btn-secondary mt-1" onClick={handleClearHistory} style={{ width: '100%', fontSize: '0.75rem' }}>🗑️ Clear</button>
          </>
        ) : (
          <div className="text-xs text-muted">No sessions yet</div>
        )}
      </aside>

      {/* ── Main ── */}
      <main className="main-content">
        <div className="hero-header">Intelligent Resume Screening<br />& Job Recommendation</div>
        <div className="hero-sub">Powered by S-BERT (all-mpnet-base-v2) · spaCy NER · TF-IDF + Random Forest · Gemini AI · Cosine Similarity</div>
        <hr className="divider" />

        {mode === 'seeker' ? <SeekerDashboard anonymize={anonymize} /> : <RecruiterDashboard anonymize={anonymize} />}

        <div className="footer">
          🧠 <strong>ResumeIQ</strong> v2.0 · S-BERT (all-mpnet-base-v2, 768-D) + TF-IDF + spaCy NER + Cosine Similarity · Research Paper Implementation
        </div>
      </main>
    </div>
  );
}
