import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const API_BASE = '/api'

const DEFAULT_SCENES_EXAMPLE = `[
  {
    "voiceover": "Your voiceover text for scene 1 goes here.",
    "prompt": "A detailed image generation prompt for scene 1."
  },
  {
    "voiceover": "Your voiceover text for scene 2 goes here.",
    "prompt": "A detailed image generation prompt for scene 2."
  }
]`

function App() {
  const [activeTab, setActiveTab] = useState('generate')
  const [models, setModels] = useState(null)
  const [error, setError] = useState('')

  // Generation settings
  const [scenesText, setScenesText] = useState('')
  const [speechProvider, setSpeechProvider] = useState('google')
  const [speechModel, setSpeechModel] = useState('gemini-2.5-pro-preview-tts')
  const [customSpeechModel, setCustomSpeechModel] = useState('')
  const [useCustomSpeechModel, setUseCustomSpeechModel] = useState(false)
  const [speechVoice, setSpeechVoice] = useState('Charon')
  const [customVoice, setCustomVoice] = useState('')
  const [useCustomVoice, setUseCustomVoice] = useState(false)
  const [imageModel, setImageModel] = useState('gemini-3.1-flash-image-preview')
  const [customImageModel, setCustomImageModel] = useState('')
  const [useCustomImageModel, setUseCustomImageModel] = useState(false)
  const [resolution, setResolution] = useState('1080p')
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [imageSize, setImageSize] = useState('512')

  // Job tracking
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [jobProgress, setJobProgress] = useState(0)
  const [jobMessage, setJobMessage] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)

  // Test panel
  const [testAudioText, setTestAudioText] = useState('Hello! This is a test of the text to speech system.')
  const [testImagePrompt, setTestImagePrompt] = useState('A beautiful sunset over the ocean with vibrant orange and purple colors.')
  const [testAudioUrl, setTestAudioUrl] = useState(null)
  const [testImageUrl, setTestImageUrl] = useState(null)
  const [testingAudio, setTestingAudio] = useState(false)
  const [testingImage, setTestingImage] = useState(false)

  const eventSourceRef = useRef(null)

  // Load models on mount
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(res => res.json())
      .then(data => setModels(data))
      .catch(() => setError('Failed to load model options. Is the backend running?'))
  }, [])

  // Update speech model & voice when provider changes
  useEffect(() => {
    if (!models) return
    const providerModels = models.speech_models[speechProvider]
    if (providerModels && providerModels.length > 0) {
      setSpeechModel(providerModels[0].value)
    }
    const providerVoices = models.speech_voices[speechProvider]
    if (providerVoices && providerVoices.length > 0) {
      setSpeechVoice(providerVoices[0].value)
    }
    setUseCustomSpeechModel(false)
    setUseCustomVoice(false)
  }, [speechProvider, models])

  const loadDefaultScenes = useCallback(() => {
    fetch(`${API_BASE}/default-scenes`)
      .then(res => res.json())
      .then(data => setScenesText(JSON.stringify(data.scenes, null, 2)))
      .catch(() => setError('Failed to load default scenes'))
  }, [])

  const parseScenes = useCallback(() => {
    try {
      const parsed = JSON.parse(scenesText)
      if (!Array.isArray(parsed)) return { valid: false, count: 0, error: 'Must be a JSON array' }
      for (const s of parsed) {
        if (!s.voiceover || !s.prompt) return { valid: false, count: parsed.length, error: 'Each scene needs "voiceover" and "prompt"' }
      }
      return { valid: true, count: parsed.length, error: null }
    } catch {
      return { valid: false, count: 0, error: 'Invalid JSON' }
    }
  }, [scenesText])

  const getActiveSpeechModel = () => useCustomSpeechModel && customSpeechModel ? customSpeechModel : speechModel
  const getActiveVoice = () => useCustomVoice && customVoice ? customVoice : speechVoice
  const getActiveImageModel = () => useCustomImageModel && customImageModel ? customImageModel : imageModel

  const startGeneration = async () => {
    const { valid, error: parseError } = parseScenes()
    if (!valid) {
      setError(parseError || 'Invalid scenes')
      return
    }

    setError('')
    setIsGenerating(true)
    setJobStatus('queued')
    setJobProgress(0)
    setJobMessage('Starting...')

    try {
      const res = await fetch(`${API_BASE}/generate-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenes: JSON.parse(scenesText),
          speech_provider: speechProvider,
          speech_model: getActiveSpeechModel(),
          speech_voice: getActiveVoice(),
          image_model: getActiveImageModel(),
          aspect_ratio: aspectRatio,
          image_size: imageSize,
          resolution,
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to start generation')
      }

      const { job_id } = await res.json()
      setJobId(job_id)

      // Connect to SSE for progress
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }

      const es = new EventSource(`${API_BASE}/progress/${job_id}`)
      eventSourceRef.current = es

      es.addEventListener('progress', (event) => {
        const data = JSON.parse(event.data)
        setJobStatus(data.status)
        setJobProgress(data.progress)
        setJobMessage(data.message)

        if (data.status === 'completed' || data.status === 'failed') {
          es.close()
          eventSourceRef.current = null
          if (data.status === 'failed') {
            setError(data.message)
          }
          setIsGenerating(false)
        }
      })

      es.onerror = () => {
        es.close()
        eventSourceRef.current = null
        // Poll status as fallback
        pollStatus(job_id)
      }
    } catch (err) {
      setError(err.message)
      setIsGenerating(false)
    }
  }

  const pollStatus = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/status/${id}`)
      const data = await res.json()
      setJobStatus(data.status)
      setJobProgress(data.progress)
      setJobMessage(data.message)
      if (data.status !== 'completed' && data.status !== 'failed') {
        setTimeout(() => pollStatus(id), 2000)
      } else {
        setIsGenerating(false)
        if (data.status === 'failed') setError(data.message)
      }
    } catch {
      setTimeout(() => pollStatus(id), 3000)
    }
  }

  const testAudio = async () => {
    if (!testAudioText.trim()) return
    setTestingAudio(true)
    setTestAudioUrl(null)
    setError('')

    try {
      const res = await fetch(`${API_BASE}/test-audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: testAudioText,
          speech_provider: speechProvider,
          speech_model: getActiveSpeechModel(),
          speech_voice: getActiveVoice(),
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Audio test failed')
      }

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setTestAudioUrl(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setTestingAudio(false)
    }
  }

  const testImage = async () => {
    if (!testImagePrompt.trim()) return
    setTestingImage(true)
    setTestImageUrl(null)
    setError('')

    try {
      const res = await fetch(`${API_BASE}/test-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: testImagePrompt,
          image_model: getActiveImageModel(),
          aspect_ratio: aspectRatio,
          image_size: imageSize,
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Image test failed')
      }

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setTestImageUrl(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setTestingImage(false)
    }
  }

  const downloadVideo = () => {
    if (jobId && jobStatus === 'completed') {
      window.open(`${API_BASE}/download/${jobId}`, '_blank')
    }
  }

  const sceneInfo = scenesText ? parseScenes() : { valid: false, count: 0, error: null }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎬 AI Video Generator</h1>
        <p>Generate AI-powered videos from scenes with customizable speech and image models</p>
      </header>

      {error && (
        <div className="alert alert-error">
          ⚠️ {error}
          <button onClick={() => setError('')} style={{ float: 'right', background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: '1rem' }}>✕</button>
        </div>
      )}

      <div className="tabs">
        <button className={`tab-btn ${activeTab === 'generate' ? 'active' : ''}`} onClick={() => setActiveTab('generate')}>
          🎥 Generate Video
        </button>
        <button className={`tab-btn ${activeTab === 'test' ? 'active' : ''}`} onClick={() => setActiveTab('test')}>
          🧪 Test Audio / Image
        </button>
      </div>

      {/* Settings Card - Shared between tabs */}
      <div className="card">
        <h2>⚙️ Model Settings</h2>

        {/* Speech Provider */}
        <div className="form-row">
          <div className="form-group">
            <label>Speech Provider</label>
            <select value={speechProvider} onChange={e => setSpeechProvider(e.target.value)}>
              {models?.speech_providers.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>
              Speech Model
              <label style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: '#6b7080' }}>
                <input
                  type="checkbox"
                  checked={useCustomSpeechModel}
                  onChange={e => setUseCustomSpeechModel(e.target.checked)}
                  style={{ marginRight: '0.25rem' }}
                />
                Custom
              </label>
            </label>
            {useCustomSpeechModel ? (
              <input
                type="text"
                placeholder="Enter custom model name..."
                value={customSpeechModel}
                onChange={e => setCustomSpeechModel(e.target.value)}
              />
            ) : (
              <select value={speechModel} onChange={e => setSpeechModel(e.target.value)}>
                {models?.speech_models[speechProvider]?.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            )}
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>
              Voice
              <label style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: '#6b7080' }}>
                <input
                  type="checkbox"
                  checked={useCustomVoice}
                  onChange={e => setUseCustomVoice(e.target.checked)}
                  style={{ marginRight: '0.25rem' }}
                />
                Custom
              </label>
            </label>
            {useCustomVoice ? (
              <input
                type="text"
                placeholder="Enter custom voice name..."
                value={customVoice}
                onChange={e => setCustomVoice(e.target.value)}
              />
            ) : (
              <select value={speechVoice} onChange={e => setSpeechVoice(e.target.value)}>
                {models?.speech_voices[speechProvider]?.map(v => (
                  <option key={v.value} value={v.value}>{v.label}</option>
                ))}
              </select>
            )}
          </div>

          <div className="form-group">
            <label>
              Image Model
              <label style={{ marginLeft: '0.5rem', fontSize: '0.75rem', color: '#6b7080' }}>
                <input
                  type="checkbox"
                  checked={useCustomImageModel}
                  onChange={e => setUseCustomImageModel(e.target.checked)}
                  style={{ marginRight: '0.25rem' }}
                />
                Custom
              </label>
            </label>
            {useCustomImageModel ? (
              <input
                type="text"
                placeholder="Enter custom image model name..."
                value={customImageModel}
                onChange={e => setCustomImageModel(e.target.value)}
              />
            ) : (
              <select value={imageModel} onChange={e => setImageModel(e.target.value)}>
                {models?.image_models?.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            )}
          </div>
        </div>

        <div className="form-row-3">
          <div className="form-group">
            <label>Resolution</label>
            <select value={resolution} onChange={e => setResolution(e.target.value)}>
              {models?.resolutions?.map(r => (
                <option key={r.value} value={r.value}>{r.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Aspect Ratio</label>
            <select value={aspectRatio} onChange={e => setAspectRatio(e.target.value)}>
              {models?.aspect_ratios?.map(a => (
                <option key={a.value} value={a.value}>{a.label}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Image Size</label>
            <select value={imageSize} onChange={e => setImageSize(e.target.value)}>
              <option value="256">256px</option>
              <option value="512">512px</option>
              <option value="1024">1024px</option>
            </select>
          </div>
        </div>
      </div>

      {/* Generate Tab */}
      {activeTab === 'generate' && (
        <>
          <div className="card">
            <div className="scene-info">
              <h2>📝 Scenes</h2>
              <div className="btn-group">
                <button className="btn btn-secondary btn-sm" onClick={loadDefaultScenes}>
                  Load Default Scenes
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => setScenesText(DEFAULT_SCENES_EXAMPLE)}>
                  Load Example Template
                </button>
              </div>
            </div>

            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <label style={{ margin: 0 }}>Paste your scenes JSON array below</label>
                {scenesText && (
                  <span className={`scene-count ${sceneInfo.valid ? 'valid' : 'invalid'}`}>
                    {sceneInfo.valid ? `✓ ${sceneInfo.count} scenes` : `✗ ${sceneInfo.error}`}
                  </span>
                )}
              </div>
              <textarea
                className="scene-editor"
                placeholder={DEFAULT_SCENES_EXAMPLE}
                value={scenesText}
                onChange={e => setScenesText(e.target.value)}
              />
              <p className="help-text">
                Each scene needs a &quot;voiceover&quot; (text for speech) and a &quot;prompt&quot; (text for image generation).
              </p>
            </div>

            <button
              className="btn btn-primary btn-block"
              disabled={!sceneInfo.valid || isGenerating}
              onClick={startGeneration}
            >
              {isGenerating ? (
                <><span className="spinner"></span> Generating...</>
              ) : (
                <>🚀 Generate Video</>
              )}
            </button>
          </div>

          {/* Progress */}
          {jobStatus && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <h2>📊 Progress</h2>
                <span className={`status-badge ${jobStatus}`}>
                  {jobStatus === 'queued' && '⏳'}
                  {jobStatus === 'processing' && '⚙️'}
                  {jobStatus === 'completed' && '✅'}
                  {jobStatus === 'failed' && '❌'}
                  {' '}{jobStatus.charAt(0).toUpperCase() + jobStatus.slice(1)}
                </span>
              </div>

              <div className="progress-bar-wrapper">
                <div
                  className={`progress-bar-fill ${jobStatus}`}
                  style={{ width: `${jobProgress}%` }}
                ></div>
              </div>

              <div className="progress-info">
                <span className="progress-message">{jobMessage}</span>
                <span className={`progress-percent ${jobStatus}`}>{jobProgress}%</span>
              </div>

              {jobStatus === 'completed' && (
                <div className="download-area">
                  <span style={{ fontSize: '2rem' }}>🎉</span>
                  <div>
                    <strong>Video Ready!</strong>
                    <p style={{ color: '#8b8fa3', fontSize: '0.85rem' }}>Your video has been generated successfully.</p>
                  </div>
                  <button className="btn btn-success" onClick={downloadVideo}>
                    ⬇️ Download Video
                  </button>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Test Tab */}
      {activeTab === 'test' && (
        <>
          <div className="card">
            <h2>🔊 Test Audio Generation</h2>
            <p style={{ color: '#8b8fa3', fontSize: '0.85rem', marginBottom: '1rem' }}>
              Test your speech model and voice settings with a sample text.
            </p>
            <div className="form-group">
              <label>Text to speak</label>
              <textarea
                value={testAudioText}
                onChange={e => setTestAudioText(e.target.value)}
                rows={3}
                style={{ minHeight: '80px' }}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={testAudio}
              disabled={testingAudio || !testAudioText.trim()}
            >
              {testingAudio ? <><span className="spinner"></span> Generating...</> : <>🔊 Test Audio</>}
            </button>

            {testAudioUrl && (
              <div className="test-result">
                <strong>✅ Audio Generated</strong>
                <audio controls src={testAudioUrl} />
              </div>
            )}
          </div>

          <div className="card">
            <h2>🖼️ Test Image Generation</h2>
            <p style={{ color: '#8b8fa3', fontSize: '0.85rem', marginBottom: '1rem' }}>
              Test your image model settings with a sample prompt.
            </p>
            <div className="form-group">
              <label>Image prompt</label>
              <textarea
                value={testImagePrompt}
                onChange={e => setTestImagePrompt(e.target.value)}
                rows={3}
                style={{ minHeight: '80px' }}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={testImage}
              disabled={testingImage || !testImagePrompt.trim()}
            >
              {testingImage ? <><span className="spinner"></span> Generating...</> : <>🖼️ Test Image</>}
            </button>

            {testImageUrl && (
              <div className="test-result">
                <strong>✅ Image Generated</strong>
                <img src={testImageUrl} alt="Generated test" />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default App
