import { useState, useEffect, useRef, useCallback } from 'react'

// Status icons (emoji for simplicity)
const ICONS = {
    rss: '📡',
    settings: '⚙️',
    download: '⬇️',
    transcribe: '🎙️',
    complete: '✅',
    skip: '⏭️',
    error: '❌',
    pending: '⏳',
}

function App() {
    // Form state
    const [rssUrl, setRssUrl] = useState('')
    const [episodeCount, setEpisodeCount] = useState(5)
    const [outputDir, setOutputDir] = useState('./transcripts')
    const [skipExisting, setSkipExisting] = useState(true)
    const [language, setLanguage] = useState('')

    // Job state
    const [jobId, setJobId] = useState(null)
    const [jobStatus, setJobStatus] = useState(null)
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [error, setError] = useState(null)
    const [episodes, setEpisodes] = useState([])

    // WebSocket ref
    const wsRef = useRef(null)

    // Connect to WebSocket for job progress
    const connectWebSocket = useCallback((id) => {
        if (wsRef.current) {
            wsRef.current.close()
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const wsUrl = `${protocol}//${window.location.host}/api/ws/progress/${id}`

        const ws = new WebSocket(wsUrl)

        ws.onopen = () => {
            console.log('WebSocket connected')
        }

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data)
                setJobStatus(data)

                // Update episodes list
                if (data.current_episode) {
                    setEpisodes(prev => {
                        const existing = prev.findIndex(
                            e => e.title === data.current_episode.episode_title
                        )
                        const updated = {
                            title: data.current_episode.episode_title,
                            status: data.current_episode.status,
                            progress: data.current_episode.progress,
                            message: data.current_episode.message,
                        }
                        if (existing >= 0) {
                            const newList = [...prev]
                            newList[existing] = updated
                            return newList
                        }
                        return [...prev, updated]
                    })
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e)
            }
        }

        ws.onerror = (error) => {
            console.error('WebSocket error:', error)
        }

        ws.onclose = () => {
            console.log('WebSocket closed')
        }

        wsRef.current = ws
    }, [])

    // Cleanup WebSocket on unmount
    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close()
            }
        }
    }, [])

    // Start transcription job
    const handleSubmit = async (e) => {
        e.preventDefault()
        setError(null)
        setIsSubmitting(true)
        setEpisodes([])

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rss_url: rssUrl,
                    episode_count: episodeCount,
                    output_dir: outputDir || null,
                    skip_existing: skipExisting,
                    language: language || null,
                }),
            })

            const text = await response.text()
            let data
            try {
                data = text ? JSON.parse(text) : {}
            } catch (parseErr) {
                throw new Error(`Server error: ${response.status} - ${text || 'No response'}`)
            }

            if (!response.ok) {
                throw new Error(data.detail || `Failed to start job (${response.status})`)
            }

            setJobId(data.job_id)
            connectWebSocket(data.job_id)
        } catch (err) {
            setError(err.message)
        } finally {
            setIsSubmitting(false)
        }
    }

    // Cancel job
    const handleCancel = async () => {
        if (!jobId) return

        try {
            await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' })
        } catch (err) {
            console.error('Failed to cancel job:', err)
        }
    }

    // Reset to start new job
    const handleReset = () => {
        setJobId(null)
        setJobStatus(null)
        setEpisodes([])
        setError(null)
        if (wsRef.current) {
            wsRef.current.close()
        }
    }

    // Calculate overall progress
    const calculateProgress = () => {
        if (!jobStatus || jobStatus.total_episodes === 0) return 0
        const done = jobStatus.completed_episodes + jobStatus.skipped_episodes + jobStatus.failed_episodes
        return (done / jobStatus.total_episodes) * 100
    }

    const isRunning = jobStatus && ['parsing', 'downloading', 'transcribing'].includes(jobStatus.status)
    const isComplete = jobStatus && ['completed', 'failed', 'cancelled'].includes(jobStatus.status)

    return (
        <div className="app-container">
            <header className="app-header">
                <h1 className="app-title">Podcast Transcript Downloader</h1>
                <p className="app-subtitle">Download and transcribe podcasts using AI</p>
            </header>

            {/* Input Form */}
            <div className="card">
                <h2 className="card-title">
                    <span className="card-title-icon">{ICONS.rss}</span>
                    Podcast RSS Feed
                </h2>

                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label className="form-label" htmlFor="rss-url">RSS Feed URL</label>
                        <input
                            id="rss-url"
                            type="url"
                            className="form-input form-input-url"
                            placeholder="https://example.com/podcast.xml"
                            value={rssUrl}
                            onChange={(e) => setRssUrl(e.target.value)}
                            disabled={isRunning}
                            required
                        />
                    </div>

                    <div className="form-row">
                        <div className="form-group">
                            <label className="form-label" htmlFor="episode-count">Episodes to Download</label>
                            <input
                                id="episode-count"
                                type="number"
                                className="form-input"
                                min="1"
                                max="50"
                                value={episodeCount}
                                onChange={(e) => setEpisodeCount(parseInt(e.target.value) || 5)}
                                disabled={isRunning}
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label" htmlFor="output-dir">Output Directory</label>
                            <input
                                id="output-dir"
                                type="text"
                                className="form-input"
                                placeholder="./transcripts"
                                value={outputDir}
                                onChange={(e) => setOutputDir(e.target.value)}
                                disabled={isRunning}
                            />
                        </div>

                        <div className="form-group">
                            <label className="form-label" htmlFor="language">Language (optional)</label>
                            <input
                                id="language"
                                type="text"
                                className="form-input"
                                placeholder="Auto-detect"
                                value={language}
                                onChange={(e) => setLanguage(e.target.value)}
                                disabled={isRunning}
                            />
                        </div>
                    </div>

                    <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <input
                            id="skip-existing"
                            type="checkbox"
                            checked={skipExisting}
                            onChange={(e) => setSkipExisting(e.target.checked)}
                            disabled={isRunning}
                        />
                        <label htmlFor="skip-existing" style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}>
                            Skip episodes with existing transcripts
                        </label>
                    </div>

                    {error && (
                        <div style={{ color: 'var(--color-accent-red)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                            {ICONS.error} {error}
                        </div>
                    )}

                    <div style={{ display: 'flex', gap: '1rem' }}>
                        {!isRunning && !isComplete && (
                            <button type="submit" className="btn btn-primary btn-full" disabled={isSubmitting || !rssUrl}>
                                {isSubmitting ? (
                                    <>
                                        <span className="spinner"></span>
                                        Starting...
                                    </>
                                ) : (
                                    <>
                                        {ICONS.transcribe} Start Transcription
                                    </>
                                )}
                            </button>
                        )}

                        {isRunning && (
                            <button type="button" className="btn btn-danger btn-full" onClick={handleCancel}>
                                Cancel Job
                            </button>
                        )}

                        {isComplete && (
                            <button type="button" className="btn btn-secondary btn-full" onClick={handleReset}>
                                Start New Job
                            </button>
                        )}
                    </div>
                </form>
            </div>

            {/* Progress Section */}
            {jobStatus && (
                <div className="card progress-section">
                    <div className="progress-header">
                        <h2 className="card-title">
                            <span className="card-title-icon">{ICONS.settings}</span>
                            {jobStatus.show_title || 'Processing...'}
                        </h2>

                        <div className={`status-badge status-${isRunning ? 'running' : isComplete ? (jobStatus.status === 'completed' ? 'success' : 'error') : 'idle'}`}>
                            {isRunning && <span className="spinner"></span>}
                            {jobStatus.status.charAt(0).toUpperCase() + jobStatus.status.slice(1)}
                        </div>
                    </div>

                    <div className="progress-stats">
                        <div className="progress-stat">
                            <span className="stat-icon stat-completed">{ICONS.complete}</span>
                            <span>{jobStatus.completed_episodes} completed</span>
                        </div>
                        <div className="progress-stat">
                            <span className="stat-icon stat-skipped">{ICONS.skip}</span>
                            <span>{jobStatus.skipped_episodes} skipped</span>
                        </div>
                        <div className="progress-stat">
                            <span className="stat-icon stat-failed">{ICONS.error}</span>
                            <span>{jobStatus.failed_episodes} failed</span>
                        </div>
                        <div className="progress-stat">
                            <span style={{ color: 'var(--color-text-muted)' }}>
                                {jobStatus.completed_episodes + jobStatus.skipped_episodes + jobStatus.failed_episodes} / {jobStatus.total_episodes} total
                            </span>
                        </div>
                    </div>

                    <div className="progress-bar-container">
                        <div
                            className="progress-bar"
                            style={{ width: `${calculateProgress()}%` }}
                        ></div>
                    </div>

                    <div className="episode-list">
                        {episodes.map((episode, idx) => (
                            <div
                                key={idx}
                                className={`episode-item status-${episode.status}`}
                            >
                                <span className="episode-status-icon">
                                    {episode.status === 'downloading' && ICONS.download}
                                    {episode.status === 'transcribing' && ICONS.transcribe}
                                    {episode.status === 'completed' && ICONS.complete}
                                    {episode.status === 'skipped' && ICONS.skip}
                                    {episode.status === 'failed' && ICONS.error}
                                    {episode.status === 'pending' && ICONS.pending}
                                </span>
                                <div className="episode-info">
                                    <div className="episode-title">{episode.title}</div>
                                    {episode.message && (
                                        <div className="episode-message">{episode.message}</div>
                                    )}
                                </div>
                                {(episode.status === 'downloading' || episode.status === 'transcribing') && (
                                    <div className="episode-progress">
                                        <div className="episode-progress-bar">
                                            <div
                                                className="episode-progress-fill"
                                                style={{ width: `${episode.progress}%` }}
                                            ></div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    {jobStatus.error && (
                        <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius-md)', color: 'var(--color-accent-red)' }}>
                            <strong>Error:</strong> {jobStatus.error}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default App
