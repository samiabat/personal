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

const DEFAULT_V2_SCENES_EXAMPLE = `[
  {
    "voiceover": "In a world consumed by noise and chaos, one traveller dared to walk the forgotten path. The ancient forest whispered secrets that only the brave could hear. And at the journey's end, a single golden light revealed the truth that had been hidden for centuries.",
    "prompts": [
      "Cinematic wide shot of a lone cloaked traveller walking along a misty mountain trail at dawn, golden hour lighting, dramatic clouds, epic fantasy landscape, 8K ultra detailed",
      "Close up of ancient twisted trees in a mystical glowing forest, shafts of golden light piercing through fog, bioluminescent mushrooms on the ground, fantasy art style",
      "Dramatic reveal of a radiant golden orb floating above a stone altar in a hidden cave, volumetric light rays, ancient runes carved into walls, cinematic lighting"
    ],
    "visual_beats": [
      { "trigger_word": "world", "effect": "zoom_in_slow", "image_index": 0 },
      { "trigger_word": "forgotten path", "effect": "hard_cut", "image_index": 1 },
      { "trigger_word": "whispered secrets", "effect": "audio_reactive_shake", "image_index": 1 },
      { "trigger_word": "golden light", "effect": "hard_cut", "image_index": 2 },
      { "trigger_word": "truth", "effect": "pop_scale", "image_index": 2 }
    ]
  }
]`

const DEFAULT_V5_SCENES_EXAMPLE = `[
  {
    "voiceover": "An elderly water bearer had two large pots. One was perfect, but the other had a tiny crack.",
    "prompt": "Vibrant Ghibli-esque anime style. An elderly Ethiopian man carrying two water pots on a mountain path. Cinematic motion.",
    "media_type": "video",
    "time_fit_strategy": "auto"
  },
  {
    "voiceover": "Every day he walked to the stream and back. The cracked pot arrived only half full, ashamed of its imperfection.",
    "prompt": "Ghibli anime style. Close-up of a cracked clay pot with water dripping on a dusty trail. Warm golden light, gentle camera pan.",
    "media_type": "video",
    "time_fit_strategy": "cinematic_slow_mo"
  }
]`

const DEFAULT_V6_SCENES_EXAMPLE = `[
  {
    "voiceover": "The ancient forest stood still as morning mist rolled through the towering trees.",
    "prompt": "Cinematic wide shot of an ancient misty forest at dawn, golden light filtering through giant trees, ethereal atmosphere, 8K.",
    "media_type": "image",
    "zoom_effect": "zoom_in",
    "focus_x": 0.5,
    "focus_y": 0.4
  },
  {
    "voiceover": "A lone figure emerged from the shadows, carrying secrets older than the stones.",
    "prompt": "Ghibli anime style. A mysterious cloaked figure walking through an ancient forest path. Cinematic motion, golden hour.",
    "media_type": "video",
    "time_fit_strategy": "auto"
  },
  {
    "voiceover": "And at last, the hidden valley revealed its breathtaking splendour.",
    "prompt": "Epic fantasy landscape of a hidden valley full of flowers and waterfalls, dramatic lighting, ultra-detailed, 8K cinematic.",
    "media_type": "image",
    "zoom_effect": "ken_burns"
  }
]`

function App() {
  const [activeTab, setActiveTab] = useState('home')
  const [models, setModels] = useState(null)
  const [error, setError] = useState('')

  // API Keys (user-provided, prioritized over .env)
  const [geminiApiKey, setGeminiApiKey] = useState('')
  const [openaiApiKey, setOpenaiApiKey] = useState('')
  const [togetherApiKey, setTogetherApiKey] = useState('')
  const [elevenlabsApiKey, setElevenlabsApiKey] = useState('')
  const [apiKeysSaved, setApiKeysSaved] = useState(false)

  // Generation settings
  const [scenesText, setScenesText] = useState('')
  const [videoVersion, setVideoVersion] = useState('v1') // 'v1', 'v2', 'v3', 'v5', or 'v6'
  const [speechProvider, setSpeechProvider] = useState('elevenlabs')
  const [speechModel, setSpeechModel] = useState('eleven_multilingual_v2')
  const [customSpeechModel, setCustomSpeechModel] = useState('')
  const [useCustomSpeechModel, setUseCustomSpeechModel] = useState(false)
  const [speechVoice, setSpeechVoice] = useState('21m00Tcm4TlvDq8ikWAM')
  const [customVoice, setCustomVoice] = useState('')
  const [useCustomVoice, setUseCustomVoice] = useState(false)
  const [imageProvider, setImageProvider] = useState('togetherai')
  const [imageModel, setImageModel] = useState('black-forest-labs/FLUX.1-schnell')
  const [customImageModel, setCustomImageModel] = useState('')
  const [useCustomImageModel, setUseCustomImageModel] = useState(false)
  const [resolution, setResolution] = useState('1080p')
  const [orientation, setOrientation] = useState('landscape')
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [imageSize, setImageSize] = useState('512')
  const [openaiImageSize, setOpenaiImageSize] = useState('1024x1024')
  const [togetheraiSize, setTogetheraiSize] = useState('1024x576')
  const [enableZoom, setEnableZoom] = useState(false)
  const [enableShake, setEnableShake] = useState(false)
  const [enableSubtitles, setEnableSubtitles] = useState(false)
  const [subtitleStyle, setSubtitleStyle] = useState('cinematic')

  // Theme
  const [darkMode, setDarkMode] = useState(true)

  // Job tracking
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [jobProgress, setJobProgress] = useState(0)
  const [jobMessage, setJobMessage] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)

  // Step-based workflow
  const [workflowStep, setWorkflowStep] = useState(0) // 0=not started, 1=generating, 2=review, 3=preparing video
  const [reviewAssets, setReviewAssets] = useState([]) // [{scene_index, prompt, image_url, ...}]
  const [regeneratingIndex, setRegeneratingIndex] = useState(null) // which scene is being regenerated
  const [assetsApproved, setAssetsApproved] = useState(false)

  // V3 Director Review
  const [directorReviewBeats, setDirectorReviewBeats] = useState([]) // zoom-related beats
  const [directorReviewIndex, setDirectorReviewIndex] = useState(0)
  const [directorFocus, setDirectorFocus] = useState(null) // {x, y} or null
  const [directorReviewDone, setDirectorReviewDone] = useState(false)

  // V5 Asset Dashboard
  const [v5UploadedVideos, setV5UploadedVideos] = useState({}) // { sceneIndex: true }
  const [v5Uploading, setV5Uploading] = useState(null) // scene index currently uploading
  const [v5CopiedIndex, setV5CopiedIndex] = useState(null) // scene index whose prompt was copied

  // V6 Asset Dashboard
  const [v6UploadedVideos, setV6UploadedVideos] = useState({}) // { sceneIndex: true } for video scenes
  const [v6Uploading, setV6Uploading] = useState(null) // scene index currently uploading
  const [v6CopiedIndex, setV6CopiedIndex] = useState(null) // scene index whose prompt was copied
  const [v6RegeneratingIndex, setV6RegeneratingIndex] = useState(null) // image scene being regenerated
  // V6 Director mode: pick focus point on image scenes
  const [v6DirectorSceneIndex, setV6DirectorSceneIndex] = useState(null) // which image scene we are editing
  const [v6DirectorFocus, setV6DirectorFocus] = useState({}) // { sceneIndex: {x, y} }
  const [v6DirectorZoom, setV6DirectorZoom] = useState({}) // { sceneIndex: zoomEffect }

  // Test panel
  const [testAudioText, setTestAudioText] = useState('Hello! This is a test of the text to speech system.')
  const [testImagePrompt, setTestImagePrompt] = useState('A beautiful sunset over the ocean with vibrant orange and purple colors.')
  const [testAudioUrl, setTestAudioUrl] = useState(null)
  const [testImageUrl, setTestImageUrl] = useState(null)
  const [testingAudio, setTestingAudio] = useState(false)
  const [testingImage, setTestingImage] = useState(false)

  // Payment / Crypto
  const [showPaymentModal, setShowPaymentModal] = useState(false)
  const [selectedPlan, setSelectedPlan] = useState(null)
  const [selectedCrypto, setSelectedCrypto] = useState(null)
  const [paymentCopied, setPaymentCopied] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // My Videos history (persisted in localStorage)
  const [myVideos, setMyVideos] = useState(() => {
    try { return JSON.parse(localStorage.getItem('omniva_my_videos') || '[]') } catch { return [] }
  })

  // Time-fit editor (shown after video is completed for V5/V6)
  const [timeFitEdits, setTimeFitEdits] = useState({}) // { sceneIndex: strategy }
  const [timeFitSaving, setTimeFitSaving] = useState(false)

  const cryptoOptions = [
    { id: 'usdc', name: 'USDC', icon: '💲', network: 'Ethereum / Polygon', address: '0x1234...your-usdc-address' },
    { id: 'usdt', name: 'USDT', icon: '💵', network: 'Ethereum / Tron', address: '0x1234...your-usdt-address' },
    { id: 'btc', name: 'Bitcoin', icon: '₿', network: 'Bitcoin Network', address: 'bc1q...your-btc-address' },
    { id: 'eth', name: 'Ethereum', icon: 'Ξ', network: 'Ethereum Network', address: '0x1234...your-eth-address' },
  ]

  const handleSelectPlan = (plan) => {
    if (plan.price === 0) return
    setSelectedPlan(plan)
    setSelectedCrypto(null)
    setPaymentCopied(false)
    setShowPaymentModal(true)
  }

  const handleCopyAddress = (address) => {
    navigator.clipboard.writeText(address).then(() => {
      setPaymentCopied(true)
      setTimeout(() => setPaymentCopied(false), 2000)
    })
  }

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

  // Update image model when image provider changes
  useEffect(() => {
    if (!models) return
    const providerModels = models.image_models?.[imageProvider]
    if (providerModels && providerModels.length > 0) {
      setImageModel(providerModels[0].value)
    }
    setUseCustomImageModel(false)
  }, [imageProvider, models])

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

      if (videoVersion === 'v5') {
        for (const s of parsed) {
          if (!s.voiceover) return { valid: false, count: parsed.length, error: 'Each V5 scene needs "voiceover"' }
          if (!s.prompt) return { valid: false, count: parsed.length, error: 'Each V5 scene needs "prompt"' }
        }
        return { valid: true, count: parsed.length, error: null }
      }

      if (videoVersion === 'v6') {
        for (const s of parsed) {
          if (!s.voiceover) return { valid: false, count: parsed.length, error: 'Each V6 scene needs "voiceover"' }
          if (!s.prompt) return { valid: false, count: parsed.length, error: 'Each V6 scene needs "prompt"' }
          const mt = s.media_type || 'image'
          if (mt !== 'image' && mt !== 'video') return { valid: false, count: parsed.length, error: 'media_type must be "image" or "video"' }
        }
        return { valid: true, count: parsed.length, error: null }
      }

      if (videoVersion === 'v2' || videoVersion === 'v3') {
        for (const s of parsed) {
          if (!s.voiceover) return { valid: false, count: parsed.length, error: 'Each V2 scene needs "voiceover"' }
          if (!Array.isArray(s.prompts) || s.prompts.length === 0) return { valid: false, count: parsed.length, error: 'Each V2 scene needs a "prompts" array with at least one prompt' }
          if (!Array.isArray(s.visual_beats) || s.visual_beats.length === 0) return { valid: false, count: parsed.length, error: 'Each V2 scene needs a "visual_beats" array' }
          for (const b of s.visual_beats) {
            if (!b.trigger_word || !b.effect || b.image_index === undefined) {
              return { valid: false, count: parsed.length, error: 'Each visual_beat needs "trigger_word", "effect", and "image_index"' }
            }
            if (b.image_index < 0 || b.image_index >= s.prompts.length) {
              return { valid: false, count: parsed.length, error: `image_index ${b.image_index} is out of range for prompts array (length ${s.prompts.length})` }
            }
          }
        }
        return { valid: true, count: parsed.length, error: null }
      }

      // V1 validation
      for (const s of parsed) {
        if (!s.voiceover || !s.prompt) return { valid: false, count: parsed.length, error: 'Each scene needs "voiceover" and "prompt"' }
      }
      return { valid: true, count: parsed.length, error: null }
    } catch {
      return { valid: false, count: 0, error: 'Invalid JSON' }
    }
  }, [scenesText, videoVersion])

  const getActiveSpeechModel = () => useCustomSpeechModel && customSpeechModel ? customSpeechModel : speechModel
  const getActiveVoice = () => useCustomVoice && customVoice ? customVoice : speechVoice
  const getActiveImageModel = () => useCustomImageModel && customImageModel ? customImageModel : imageModel

  const getTogetheraiDimensions = () => {
    const [w, h] = togetheraiSize.split('x').map(Number)
    return { width: w, height: h }
  }

  const connectSSE = (id, onAssetsReady) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    const es = new EventSource(`${API_BASE}/progress/${id}`)
    eventSourceRef.current = es

    es.addEventListener('progress', (event) => {
      const data = JSON.parse(event.data)
      setJobStatus(data.status)
      setJobProgress(data.progress)
      setJobMessage(data.message)

      if (data.status === 'assets_ready') {
        es.close()
        eventSourceRef.current = null
        setIsGenerating(false)
        if (onAssetsReady) onAssetsReady()
      } else if (data.status === 'completed' || data.status === 'failed') {
        es.close()
        eventSourceRef.current = null
        if (data.status === 'failed') {
          setError(data.message)
        }
        if (data.status === 'completed') {
          setWorkflowStep(4) // completed
          // Save to My Videos history
          setMyVideos(prev => {
            const entry = {
              jobId: id,
              timestamp: Date.now(),
              version: videoVersion,
            }
            const updated = [entry, ...prev.filter(v => v.jobId !== id)].slice(0, 50)
            try { localStorage.setItem('omniva_my_videos', JSON.stringify(updated)) } catch { /* ignore storage errors */ }
            return updated
          })
          // Initialise time-fit editor from current request scenes
          setTimeFitEdits({})
        }
        setIsGenerating(false)
      }
    })

    es.onerror = () => {
      es.close()
      eventSourceRef.current = null
      pollStatus(id)
    }
  }

  const loadReviewAssets = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/job-assets/${id}`)
      if (!res.ok) throw new Error('Failed to load assets')
      const data = await res.json()
      setReviewAssets(data.assets)
      setWorkflowStep(2) // move to review step
    } catch (err) {
      setError(err.message)
    }
  }

  const regenerateV2Image = async (sceneIndex, promptIndex) => {
    if (!jobId) return
    setRegeneratingIndex(`${sceneIndex}_${promptIndex}`)
    setError('')

    try {
      const res = await fetch(`${API_BASE}/regenerate-v2-image/${jobId}/${sceneIndex}/${promptIndex}`, {
        method: 'POST',
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Regeneration failed')
      }

      // Update the specific image URL with cache-busting to force reload
      setReviewAssets(prev => prev.map((scene, sIdx) =>
        sIdx === sceneIndex
          ? {
              ...scene,
              images: scene.images.map((img, pIdx) =>
                pIdx === promptIndex
                  ? { ...img, image_url: `${API_BASE}/v2-scene-image/${jobId}/${sceneIndex}/${promptIndex}?t=${Date.now()}` }
                  : img
              ),
            }
          : scene
      ))
    } catch (err) {
      setError(err.message)
    } finally {
      setRegeneratingIndex(null)
    }
  }

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
    setWorkflowStep(1)
    setAssetsApproved(false)
    setReviewAssets([])
    setDirectorReviewBeats([])
    setDirectorReviewIndex(0)
    setDirectorFocus(null)
    setDirectorReviewDone(false)
    setV6UploadedVideos({})
    setV6DirectorSceneIndex(null)
    setV6DirectorFocus({})
    setV6DirectorZoom({})

    const togDims = getTogetheraiDimensions()
    const parsedScenes = JSON.parse(scenesText)

    const body = {
      version: videoVersion,
      speech_provider: speechProvider,
      speech_model: getActiveSpeechModel(),
      speech_voice: getActiveVoice(),
      image_provider: imageProvider,
      image_model: getActiveImageModel(),
      aspect_ratio: aspectRatio,
      image_size: imageSize,
      openai_image_size: openaiImageSize,
      togetherai_width: togDims.width,
      togetherai_height: togDims.height,
      resolution,
      orientation,
      enable_zoom: enableZoom,
      enable_shake: enableShake,
      enable_subtitles: enableSubtitles,
      subtitle_style: subtitleStyle,
      gemini_api_key: geminiApiKey,
      openai_api_key: openaiApiKey,
      together_api_key: togetherApiKey,
      elevenlabs_api_key: elevenlabsApiKey,
    }

    if (videoVersion === 'v5') {
      body.v5_scenes = parsedScenes
    } else if (videoVersion === 'v6') {
      body.v6_scenes = parsedScenes
    } else if (videoVersion === 'v2' || videoVersion === 'v3') {
      body.v2_scenes = parsedScenes
    } else {
      body.scenes = parsedScenes
    }

    try {
      const res = await fetch(`${API_BASE}/generate-assets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to start generation')
      }

      const { job_id } = await res.json()
      setJobId(job_id)
      connectSSE(job_id, () => loadReviewAssets(job_id))
    } catch (err) {
      setError(err.message)
      setIsGenerating(false)
      setWorkflowStep(0)
    }
  }

  const regenerateImage = async (sceneIndex) => {
    if (!jobId) return
    setRegeneratingIndex(sceneIndex)
    setError('')

    try {
      const res = await fetch(`${API_BASE}/regenerate-image/${jobId}/${sceneIndex}`, {
        method: 'POST',
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Regeneration failed')
      }

      // Refresh the asset list with cache-busting
      setReviewAssets(prev => prev.map((a, i) =>
        i === sceneIndex ? { ...a, image_url: `${API_BASE}/scene-image/${jobId}/${sceneIndex}?t=${Date.now()}` } : a
      ))
    } catch (err) {
      setError(err.message)
    } finally {
      setRegeneratingIndex(null)
    }
  }

  const approveAssets = async () => {
    if (!jobId) return
    setError('')

    try {
      const res = await fetch(`${API_BASE}/approve-assets/${jobId}`, {
        method: 'POST',
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Approval failed')
      }

      setAssetsApproved(true)
    } catch (err) {
      setError(err.message)
    }
  }

  // ── V5 Asset Dashboard helpers ──

  const v5CopyPrompt = async (prompt, sceneIndex) => {
    try {
      await navigator.clipboard.writeText(prompt)
      setV5CopiedIndex(sceneIndex)
      setTimeout(() => setV5CopiedIndex(null), 2000)
    } catch {
      // Fallback for older browsers
      const ta = document.createElement('textarea')
      ta.value = prompt
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      setV5CopiedIndex(sceneIndex)
      setTimeout(() => setV5CopiedIndex(null), 2000)
    }
  }

  const v5UploadVideo = async (sceneIndex, file) => {
    if (!jobId || !file) return
    setV5Uploading(sceneIndex)
    setError('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/v5-upload-video/${jobId}/${sceneIndex}`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Upload failed')
      }

      setV5UploadedVideos(prev => ({ ...prev, [sceneIndex]: true }))

      // Refresh review assets to reflect the new video
      setReviewAssets(prev => prev.map((a, i) =>
        i === sceneIndex ? { ...a, has_video: true, video_url: `${API_BASE}/v5-scene-video/${jobId}/${sceneIndex}?t=${Date.now()}` } : a
      ))
    } catch (err) {
      setError(err.message)
    } finally {
      setV5Uploading(null)
    }
  }

  const v5AllVideosUploaded = () => {
    if (videoVersion !== 'v5') return true
    return reviewAssets.length > 0 && reviewAssets.every((a, i) => a.has_video || v5UploadedVideos[i])
  }

  // ── V6 Asset Dashboard helpers ──

  const v6CopyPrompt = async (prompt, sceneIndex) => {
    try {
      await navigator.clipboard.writeText(prompt)
      setV6CopiedIndex(sceneIndex)
      setTimeout(() => setV6CopiedIndex(null), 2000)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = prompt
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      setV6CopiedIndex(sceneIndex)
      setTimeout(() => setV6CopiedIndex(null), 2000)
    }
  }

  const v6UploadVideo = async (sceneIndex, file) => {
    if (!jobId || !file) return
    setV6Uploading(sceneIndex)
    setError('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/v6-upload-video/${jobId}/${sceneIndex}`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Upload failed')
      }
      setV6UploadedVideos(prev => ({ ...prev, [sceneIndex]: true }))
      setReviewAssets(prev => prev.map((a, i) =>
        i === sceneIndex ? { ...a, has_video: true, video_url: `${API_BASE}/v6-scene-video/${jobId}/${sceneIndex}?t=${Date.now()}` } : a
      ))
    } catch (err) {
      setError(err.message)
    } finally {
      setV6Uploading(null)
    }
  }

  const v6RegenerateImage = async (sceneIndex) => {
    if (!jobId) return
    setV6RegeneratingIndex(sceneIndex)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/v6-regenerate-image/${jobId}/${sceneIndex}`, { method: 'POST' })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Regeneration failed')
      }
      setReviewAssets(prev => prev.map((a, i) =>
        i === sceneIndex ? { ...a, image_url: `${API_BASE}/v6-scene-image/${jobId}/${sceneIndex}?t=${Date.now()}` } : a
      ))
    } catch (err) {
      setError(err.message)
    } finally {
      setV6RegeneratingIndex(null)
    }
  }

  const v6SaveFocusPoint = async (sceneIndex, x, y, zoomEffect) => {
    if (!jobId) return
    try {
      const body = { scene_index: sceneIndex, focus_x: x, focus_y: y }
      if (zoomEffect !== undefined) body.zoom_effect = zoomEffect
      await fetch(`${API_BASE}/v6-update-scene/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setV6DirectorFocus(prev => ({ ...prev, [sceneIndex]: { x, y } }))
      if (zoomEffect !== undefined) {
        setV6DirectorZoom(prev => ({ ...prev, [sceneIndex]: zoomEffect }))
        setReviewAssets(prev => prev.map((a, i) =>
          i === sceneIndex ? { ...a, zoom_effect: zoomEffect, focus_x: x, focus_y: y } : a
        ))
      }
    } catch (err) {
      setError(err.message)
    }
  }

  const v6AllVideoScenesUploaded = () => {
    if (videoVersion !== 'v6') return true
    return reviewAssets.length > 0 && reviewAssets.every((a, i) => {
      if (a.media_type !== 'video') return true
      return a.has_video || v6UploadedVideos[i]
    })
  }

  const prepareVideo = async () => {
    if (!jobId) return
    setError('')
    setIsGenerating(true)
    setJobStatus('queued')
    setJobProgress(0)
    setJobMessage('Preparing video...')
    setWorkflowStep(3)

    try {
      const res = await fetch(`${API_BASE}/prepare-video/${jobId}`, {
        method: 'POST',
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to start video preparation')
      }

      connectSSE(jobId)
    } catch (err) {
      setError(err.message)
      setIsGenerating(false)
    }
  }

  // --- V3 Director Review helpers ---
  const ZOOM_EFFECTS = ['zoom_in_slow', 'zoom_out_slow', 'pop_scale']

  const initDirectorReview = () => {
    // Collect all zoom-related beats across all scenes
    const beats = []
    for (let sIdx = 0; sIdx < reviewAssets.length; sIdx++) {
      const scene = reviewAssets[sIdx]
      const vBeats = scene.visual_beats || []
      for (let bIdx = 0; bIdx < vBeats.length; bIdx++) {
        const b = vBeats[bIdx]
        if (ZOOM_EFFECTS.includes(b.effect)) {
          beats.push({
            sceneIndex: sIdx,
            beatIndex: bIdx,
            triggerWord: b.trigger_word,
            effect: b.effect,
            imageIndex: b.image_index,
            imageUrl: (scene.images || [])[b.image_index]?.image_url || '',
          })
        }
      }
    }
    if (beats.length === 0) {
      // No zoom beats — skip director review entirely
      setDirectorReviewDone(true)
      return
    }
    setDirectorReviewBeats(beats)
    setDirectorReviewIndex(0)
    setDirectorFocus(null)
    setDirectorReviewDone(false)
  }

  const handleDirectorImageClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    setDirectorFocus({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) })
  }

  const acceptFocusPoint = async () => {
    if (!jobId || directorReviewIndex >= directorReviewBeats.length) return
    const beat = directorReviewBeats[directorReviewIndex]
    const fx = directorFocus ? directorFocus.x : 0.5
    const fy = directorFocus ? directorFocus.y : 0.5

    try {
      const res = await fetch(`${API_BASE}/update-focus-point/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scene_index: beat.sceneIndex,
          beat_index: beat.beatIndex,
          focus_x: fx,
          focus_y: fy,
        }),
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to save focus point')
      }
    } catch (err) {
      setError(err.message)
      return
    }

    advanceDirectorReview()
  }

  const skipFocusPoint = async () => {
    // Use centre (0.5, 0.5)
    if (!jobId || directorReviewIndex >= directorReviewBeats.length) return
    const beat = directorReviewBeats[directorReviewIndex]

    try {
      await fetch(`${API_BASE}/update-focus-point/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scene_index: beat.sceneIndex,
          beat_index: beat.beatIndex,
          focus_x: 0.5,
          focus_y: 0.5,
        }),
      })
    } catch (err) { console.error('Skip focus point:', err) }

    advanceDirectorReview()
  }

  const advanceDirectorReview = () => {
    const next = directorReviewIndex + 1
    if (next >= directorReviewBeats.length) {
      setDirectorReviewDone(true)
    } else {
      setDirectorReviewIndex(next)
      setDirectorFocus(null)
    }
  }

  const retryGeneration = async () => {
    if (!jobId) return
    setError('')
    setIsGenerating(true)
    setJobStatus('queued')
    setJobProgress(0)
    setJobMessage('Retrying...')

    try {
      const res = await fetch(`${API_BASE}/retry/${jobId}`, {
        method: 'POST',
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to retry')
      }

      connectSSE(jobId)
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
      if (data.status === 'assets_ready') {
        setIsGenerating(false)
        loadReviewAssets(id)
      } else if (data.status !== 'completed' && data.status !== 'failed') {
        setTimeout(() => pollStatus(id), 2000)
      } else {
        setIsGenerating(false)
        if (data.status === 'failed') setError(data.message)
        if (data.status === 'completed') {
          setWorkflowStep(4)
          setMyVideos(prev => {
            const entry = { jobId: id, timestamp: Date.now(), version: videoVersion }
            const updated = [entry, ...prev.filter(v => v.jobId !== id)].slice(0, 50)
            try { localStorage.setItem('omniva_my_videos', JSON.stringify(updated)) } catch { /* ignore storage errors */ }
            return updated
          })
          setTimeFitEdits({})
        }
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
          gemini_api_key: geminiApiKey,
          openai_api_key: openaiApiKey,
          elevenlabs_api_key: elevenlabsApiKey,
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

    const togDims = getTogetheraiDimensions()

    try {
      const res = await fetch(`${API_BASE}/test-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: testImagePrompt,
          image_provider: imageProvider,
          image_model: getActiveImageModel(),
          aspect_ratio: aspectRatio,
          image_size: imageSize,
          openai_image_size: openaiImageSize,
          togetherai_width: togDims.width,
          togetherai_height: togDims.height,
          gemini_api_key: geminiApiKey,
          openai_api_key: openaiApiKey,
          together_api_key: togetherApiKey,
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

  // Speech settings block (reusable)
  const renderSpeechSettings = () => (
    <>
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
            <label className="custom-toggle">
              <input type="checkbox" checked={useCustomSpeechModel} onChange={e => setUseCustomSpeechModel(e.target.checked)} />
              Custom
            </label>
          </label>
          {useCustomSpeechModel ? (
            <input type="text" placeholder="Enter custom model name..." value={customSpeechModel} onChange={e => setCustomSpeechModel(e.target.value)} />
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
            <label className="custom-toggle">
              <input type="checkbox" checked={useCustomVoice} onChange={e => setUseCustomVoice(e.target.checked)} />
              Custom
            </label>
          </label>
          {useCustomVoice ? (
            <input type="text" placeholder="Enter custom voice name..." value={customVoice} onChange={e => setCustomVoice(e.target.value)} />
          ) : (
            <select value={speechVoice} onChange={e => setSpeechVoice(e.target.value)}>
              {models?.speech_voices[speechProvider]?.map(v => (
                <option key={v.value} value={v.value}>{v.label}</option>
              ))}
            </select>
          )}
        </div>
      </div>
    </>
  )

  // Image settings block (reusable)
  const renderImageSettings = () => (
    <>
      <div className="form-row">
        <div className="form-group">
          <label>Image Provider</label>
          <select value={imageProvider} onChange={e => setImageProvider(e.target.value)}>
            {models?.image_providers?.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>
            Image Model
            <label className="custom-toggle">
              <input type="checkbox" checked={useCustomImageModel} onChange={e => setUseCustomImageModel(e.target.checked)} />
              Custom
            </label>
          </label>
          {useCustomImageModel ? (
            <input type="text" placeholder="Enter custom image model name..." value={customImageModel} onChange={e => setCustomImageModel(e.target.value)} />
          ) : (
            <select value={imageModel} onChange={e => setImageModel(e.target.value)}>
              {models?.image_models?.[imageProvider]?.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          )}
        </div>
      </div>
      {imageProvider === 'gemini' && (
        <div className="form-row">
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
      )}
      {imageProvider === 'openai' && (
        <div className="form-row">
          <div className="form-group">
            <label>Image Size</label>
            <select value={openaiImageSize} onChange={e => setOpenaiImageSize(e.target.value)}>
              {models?.openai_image_sizes?.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>
        </div>
      )}
      {imageProvider === 'togetherai' && (
        <div className="form-row">
          <div className="form-group">
            <label>Image Dimensions</label>
            <select value={togetheraiSize} onChange={e => setTogetheraiSize(e.target.value)}>
              {models?.togetherai_sizes?.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>
        </div>
      )}
    </>
  )

  return (
    <div className={`app-shell ${darkMode ? 'theme-dark' : 'theme-light'}`}>
      {/* Top Navigation Bar */}
      <nav className="top-nav">
        <div className="nav-inner">
          <div className="nav-brand" onClick={() => setActiveTab('home')}>
            <span className="nav-logo-mark">▶</span>
            <span className="nav-logo-text">Omniva <span className="nav-logo-ai">Video Forge</span></span>
          </div>
          <button className="nav-mobile-toggle" onClick={() => setMobileMenuOpen(prev => !prev)} aria-label="Toggle menu">
            <span className={`hamburger ${mobileMenuOpen ? 'open' : ''}`} />
          </button>
          <div className={`nav-links ${mobileMenuOpen ? 'nav-links-open' : ''}`}>
            {[
              { key: 'home', label: 'Home' },
              { key: 'create', label: 'Create' },
              { key: 'myvideos', label: 'My Videos' },
              { key: 'test', label: 'Test Lab' },
              { key: 'templates', label: 'Templates' },
              { key: 'pricing', label: 'Pricing' },
              { key: 'settings', label: 'Settings' },
            ].map(item => (
              <button
                key={item.key}
                className={`nav-link ${activeTab === item.key ? 'active' : ''}`}
                onClick={() => { setActiveTab(item.key); setMobileMenuOpen(false) }}
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="nav-actions">
            <button className="theme-toggle-btn" onClick={() => setDarkMode(prev => !prev)} title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}>
              {darkMode ? '☀️' : '🌙'}
            </button>
            <span className="nav-avatar">U</span>
          </div>
        </div>
      </nav>

      <main className="app-main">
        {error && (
          <div className="alert alert-error">
            <span>⚠️ {error}</span>
            <button onClick={() => setError('')} className="alert-close">✕</button>
          </div>
        )}

        {/* ─── Home / Landing Page ─── */}
        {activeTab === 'home' && (
          <div className="page-content landing-page">
            {/* Hero Section */}
            <section className="hero-section">
              <div className="hero-badge">🚀 AI-Powered Video Generation Platform</div>
              <h1 className="hero-title">
                Create Stunning Videos <br />
                <span className="hero-gradient">with AI in Minutes</span>
              </h1>
              <p className="hero-description">
                Transform your ideas into professional videos using multiple AI providers.
                Generate voiceovers, images, and fully assembled videos — all from simple text prompts.
              </p>
              <div className="hero-actions">
                <button className="btn btn-primary btn-lg btn-glow" onClick={() => setActiveTab('create')}>
                  🎬 Start Creating — Free
                </button>
                <button className="btn btn-outline btn-lg" onClick={() => setActiveTab('pricing')}>
                  View Pricing
                </button>
              </div>
              <div className="hero-stats-row">
                <div className="hero-stat">
                  <span className="hero-stat-value">3+</span>
                  <span className="hero-stat-label">AI Providers</span>
                </div>
                <div className="hero-stat-divider"></div>
                <div className="hero-stat">
                  <span className="hero-stat-value">15+</span>
                  <span className="hero-stat-label">AI Models</span>
                </div>
                <div className="hero-stat-divider"></div>
                <div className="hero-stat">
                  <span className="hero-stat-value">4K</span>
                  <span className="hero-stat-label">Max Resolution</span>
                </div>
                <div className="hero-stat-divider"></div>
                <div className="hero-stat">
                  <span className="hero-stat-value">BYO</span>
                  <span className="hero-stat-label">API Keys</span>
                </div>
              </div>
            </section>

            {/* Features Grid */}
            <section className="landing-section">
              <div className="section-header-center">
                <h2>Everything You Need to Create Professional Videos</h2>
                <p>A complete AI video production pipeline — from script to screen.</p>
              </div>
              <div className="features-grid">
                {[
                  { icon: '🎙️', title: 'AI Voiceover', desc: 'Natural-sounding text-to-speech using Google Gemini, OpenAI, or ElevenLabs. Choose from dozens of voices and models.' },
                  { icon: '🖼️', title: 'AI Image Generation', desc: 'Generate stunning visuals with Gemini, OpenAI DALL·E, or Together AI FLUX models.' },
                  { icon: '🎬', title: 'Automated Assembly', desc: 'Audio and images are automatically assembled into polished videos with transitions and effects.' },
                  { icon: '🔍', title: 'Ken Burns Effects', desc: 'Add cinematic zoom and pan effects to bring static images to life in your videos.' },
                  { icon: '📱', title: 'Portrait & Landscape', desc: 'Create videos in any orientation — landscape for YouTube, portrait for Shorts, Reels & TikTok.' },
                  { icon: '🔑', title: 'Bring Your Own Keys', desc: 'Use your own API keys for complete control over costs, or use our hosted keys to get started instantly.' },
                ].map((f, i) => (
                  <div className="feature-card" key={i}>
                    <div className="feature-icon">{f.icon}</div>
                    <h3>{f.title}</h3>
                    <p>{f.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* How It Works */}
            <section className="landing-section">
              <div className="section-header-center">
                <h2>How It Works</h2>
                <p>Three simple steps to create your AI-powered video.</p>
              </div>
              <div className="how-it-works-grid">
                <div className="how-step">
                  <div className="how-step-number">1</div>
                  <h3>Define Your Scenes</h3>
                  <p>Write your voiceover text and image prompts for each scene. Use our templates or create custom content from scratch.</p>
                </div>
                <div className="how-step-connector">
                  <span className="how-step-arrow">→</span>
                </div>
                <div className="how-step">
                  <div className="how-step-number">2</div>
                  <h3>Review & Refine</h3>
                  <p>Preview generated audio and images. Regenerate any scene that doesn&apos;t look right until you&apos;re satisfied.</p>
                </div>
                <div className="how-step-connector">
                  <span className="how-step-arrow">→</span>
                </div>
                <div className="how-step">
                  <div className="how-step-number">3</div>
                  <h3>Download Your Video</h3>
                  <p>Approve your assets and we&apos;ll assemble a polished video ready to share on any platform.</p>
                </div>
              </div>
            </section>

            {/* Providers */}
            <section className="landing-section">
              <div className="section-header-center">
                <h2>Powered by Leading AI Providers</h2>
                <p>Choose the best model for your needs — switch providers anytime.</p>
              </div>
              <div className="providers-grid">
                <div className="provider-card">
                  <div className="provider-icon">✦</div>
                  <h3>Google Gemini</h3>
                  <p>State-of-the-art TTS and image generation with Gemini 2.5 Pro, Flash, and Imagen 3.0 models.</p>
                  <div className="provider-tags">
                    <span className="provider-tag">Speech</span>
                    <span className="provider-tag">Images</span>
                  </div>
                </div>
                <div className="provider-card">
                  <div className="provider-icon">◆</div>
                  <h3>OpenAI</h3>
                  <p>Premium TTS voices with GPT-4o Mini and high-quality image generation with DALL·E and GPT Image models.</p>
                  <div className="provider-tags">
                    <span className="provider-tag">Speech</span>
                    <span className="provider-tag">Images</span>
                  </div>
                </div>
                <div className="provider-card">
                  <div className="provider-icon">▲</div>
                  <h3>Together AI</h3>
                  <p>Open-source image models including FLUX.1, Stable Diffusion XL, and DreamShaper for fast, affordable generation.</p>
                  <div className="provider-tags">
                    <span className="provider-tag provider-tag-default">Default Images</span>
                  </div>
                </div>
                <div className="provider-card">
                  <div className="provider-icon">🎙️</div>
                  <h3>ElevenLabs</h3>
                  <p>Ultra-realistic AI voices with multilingual support, turbo, and flash models for the most natural-sounding narration.</p>
                  <div className="provider-tags">
                    <span className="provider-tag provider-tag-default">Default Speech</span>
                  </div>
                </div>
              </div>
            </section>

            {/* CTA */}
            <section className="landing-section cta-section">
              <div className="cta-card">
                <h2>Ready to Create Your First Video?</h2>
                <p>Get started for free — no credit card required. Bring your own API keys or use our hosted service.</p>
                <div className="hero-actions">
                  <button className="btn btn-primary btn-lg btn-glow" onClick={() => setActiveTab('create')}>
                    🎬 Create Your First Video
                  </button>
                  <button className="btn btn-outline btn-lg" onClick={() => setActiveTab('settings')}>
                    🔑 Configure API Keys
                  </button>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ─── Create Video Page ─── */}
        {activeTab === 'create' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Create Video</h1>
              <p className="page-subtitle">Configure settings and generate AI-powered videos</p>
            </div>

            {/* Full Model Settings Card */}
            <div className="card">
              <div className="card-header">
                <h2>⚙️ Model Settings</h2>
                <span className="card-badge">Configuration</span>
              </div>

              <div className="settings-section">
                <h3 className="section-title">🔊 Speech Settings</h3>
                {renderSpeechSettings()}
              </div>

              <div className="settings-section">
                <h3 className="section-title">🖼️ Image Settings</h3>
                {renderImageSettings()}
              </div>

              <div className="settings-section">
                <h3 className="section-title">🎞️ Video Settings</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Output Resolution</label>
                    <select value={resolution} onChange={e => setResolution(e.target.value)}>
                      {models?.resolutions?.map(r => (
                        <option key={r.value} value={r.value}>{r.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Orientation</label>
                    <select value={orientation} onChange={e => setOrientation(e.target.value)}>
                      {models?.orientations?.map(o => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                    <p className="help-text">
                      {orientation === 'portrait' ? 'Vertical video for YouTube Shorts, Instagram Reels & TikTok' : 'Standard widescreen video'}
                    </p>
                  </div>
                </div>
                <div className="effects-grid">
                  <div className="effect-card">
                    <div className="effect-header">
                      <span className="effect-icon">🔍</span>
                      <span className="effect-title">Gentle Zoom</span>
                      <span className="help-tooltip" title="Adds a subtle slow zoom animation to each scene image.">ℹ️</span>
                    </div>
                    <p className="effect-desc">Slow cinematic zoom in/out on each scene</p>
                    <label className="switch-container">
                      <input type="checkbox" checked={enableZoom} onChange={e => setEnableZoom(e.target.checked)} />
                      <span className="switch-slider"></span>
                      <span className="switch-label">{enableZoom ? 'On' : 'Off'}</span>
                    </label>
                  </div>
                  <div className="effect-card">
                    <div className="effect-header">
                      <span className="effect-icon">↔️</span>
                      <span className="effect-title">Gentle Pan</span>
                      <span className="help-tooltip" title="Adds a very subtle panning motion to each scene image.">ℹ️</span>
                    </div>
                    <p className="effect-desc">Subtle horizontal &amp; vertical drift</p>
                    <label className="switch-container">
                      <input type="checkbox" checked={enableShake} onChange={e => setEnableShake(e.target.checked)} />
                      <span className="switch-slider"></span>
                      <span className="switch-label">{enableShake ? 'On' : 'Off'}</span>
                    </label>
                  </div>
                  <div className="effect-card">
                    <div className="effect-header">
                      <span className="effect-icon">✨</span>
                      <span className="effect-title">Fancy Subtitles</span>
                      <span className="help-tooltip" title="Overlay stylish text subtitles on your video using word-level timestamps from the voiceover.">ℹ️</span>
                    </div>
                    <p className="effect-desc">Add designer-style text overlays on the video</p>
                    <label className="switch-container">
                      <input type="checkbox" checked={enableSubtitles} onChange={e => setEnableSubtitles(e.target.checked)} />
                      <span className="switch-slider"></span>
                      <span className="switch-label">{enableSubtitles ? 'On' : 'Off'}</span>
                    </label>
                    {enableSubtitles && (
                      <div style={{ marginTop: '0.5rem' }}>
                        <label className="field-label" style={{ fontSize: '0.75rem', marginBottom: '0.25rem', display: 'block' }}>Style</label>
                        <select
                          className="select"
                          value={subtitleStyle}
                          onChange={e => setSubtitleStyle(e.target.value)}
                          style={{ width: '100%', fontSize: '0.8rem', padding: '0.3rem' }}
                        >
                          <option value="cinematic">🎬 Cinematic</option>
                          <option value="minimal">📝 Minimal</option>
                          <option value="typewriter">⌨️ Typewriter</option>
                        </select>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Scenes Card */}
            <div className="card">
              <div className="card-header">
                <h2>📝 Scenes</h2>
                <div className="btn-group">
                  <div className="version-toggle" style={{ display: 'flex', gap: '0.25rem', background: darkMode ? '#1e1e2e' : '#e8e8e8', borderRadius: '0.5rem', padding: '0.2rem' }}>
                    <button
                      className={`btn btn-sm ${videoVersion === 'v1' ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => { setVideoVersion('v1'); setScenesText('') }}
                      style={{ borderRadius: '0.4rem', minWidth: '3rem' }}
                    >
                      V1
                    </button>
                    <button
                      className={`btn btn-sm ${videoVersion === 'v2' ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => { setVideoVersion('v2'); setScenesText('') }}
                      style={{ borderRadius: '0.4rem', minWidth: '3rem' }}
                    >
                      V2
                    </button>
                    <button
                      className={`btn btn-sm ${videoVersion === 'v3' ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => { setVideoVersion('v3'); setScenesText('') }}
                      style={{ borderRadius: '0.4rem', minWidth: '3rem' }}
                    >
                      V3
                    </button>
                    <button
                      className={`btn btn-sm ${videoVersion === 'v5' ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => { setVideoVersion('v5'); setScenesText(''); setV5UploadedVideos({}); setV5CopiedIndex(null) }}
                      style={{ borderRadius: '0.4rem', minWidth: '3rem' }}
                    >
                      V5
                    </button>
                    <button
                      className={`btn btn-sm ${videoVersion === 'v6' ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => { setVideoVersion('v6'); setScenesText(''); setV6UploadedVideos({}); setV6CopiedIndex(null); setV6DirectorSceneIndex(null); setV6DirectorFocus({}); setV6DirectorZoom({}) }}
                      style={{ borderRadius: '0.4rem', minWidth: '3rem' }}
                    >
                      V6
                    </button>
                  </div>
                  {videoVersion === 'v1' && (
                    <>
                      <button className="btn btn-secondary btn-sm" onClick={loadDefaultScenes}>
                        Load Default Scenes
                      </button>
                      <button className="btn btn-secondary btn-sm" onClick={() => setScenesText(DEFAULT_SCENES_EXAMPLE)}>
                        Load Example Template
                      </button>
                    </>
                  )}
                  {(videoVersion === 'v2' || videoVersion === 'v3') && (
                    <button className="btn btn-secondary btn-sm" onClick={() => setScenesText(DEFAULT_V2_SCENES_EXAMPLE)}>
                      Load {videoVersion.toUpperCase()} Example
                    </button>
                  )}
                  {videoVersion === 'v5' && (
                    <button className="btn btn-secondary btn-sm" onClick={() => setScenesText(DEFAULT_V5_SCENES_EXAMPLE)}>
                      Load V5 Example
                    </button>
                  )}
                  {videoVersion === 'v6' && (
                    <button className="btn btn-secondary btn-sm" onClick={() => setScenesText(DEFAULT_V6_SCENES_EXAMPLE)}>
                      Load V6 Example
                    </button>
                  )}
                </div>
              </div>

              {(videoVersion === 'v2' || videoVersion === 'v3') && (
                <div className="v2-info-banner" style={{
                  background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08))',
                  border: '1px solid rgba(99,102,241,0.25)',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1rem',
                  marginBottom: '1rem',
                  fontSize: '0.85rem',
                  color: darkMode ? '#a5b4fc' : '#4338ca',
                }}>
                  <strong>{videoVersion === 'v3' ? 'V3 Director Mode' : 'V2 Grouped Scenes'}:</strong> One voiceover maps to multiple images via visual beats.
                  <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', lineHeight: '1.5' }}>
                    <strong>Effects:</strong>{' '}
                    <code>zoom_in_slow</code> (gradual zoom in),{' '}
                    <code>zoom_out_slow</code> (gradual zoom out),{' '}
                    <code>audio_reactive_shake</code> (jitter at audio peak),{' '}
                    <code>hard_cut</code> (instant switch to image),{' '}
                    <code>pop_scale</code> (quick scale burst).
                  </div>
                  <div style={{ marginTop: '0.25rem', fontSize: '0.8rem', lineHeight: '1.5' }}>
                    <strong>Color grades</strong> (optional):{' '}
                    <code>dark</code>, <code>warm</code>, <code>cool</code>, <code>high_contrast</code>.
                  </div>
                  {videoVersion === 'v3' && (
                    <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', lineHeight: '1.5', borderTop: '1px solid rgba(99,102,241,0.2)', paddingTop: '0.4rem' }}>
                      🎯 <strong>Director Review:</strong> After approving assets, you can click on images to set a custom focus point for zoom &amp; pop effects.
                    </div>
                  )}
                </div>
              )}

              {videoVersion === 'v5' && (
                <div className="v5-info-banner" style={{
                  background: 'linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.08))',
                  border: '1px solid rgba(16,185,129,0.25)',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1rem',
                  marginBottom: '1rem',
                  fontSize: '0.85rem',
                  color: darkMode ? '#6ee7b7' : '#047857',
                }}>
                  <strong>🎬 V5 Video Clip Mode:</strong> Upload AI-generated video clips (e.g. from Grok) for each scene.
                  <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', lineHeight: '1.5' }}>
                    <strong>Workflow:</strong> Generate voiceover → Copy prompt → Generate 6s clip in Grok → Upload clip → Render
                  </div>
                  <div style={{ marginTop: '0.25rem', fontSize: '0.8rem', lineHeight: '1.5' }}>
                    <strong>Time-fit strategies:</strong>{' '}
                    <code>auto</code> (smart select),{' '}
                    <code>trim</code> (audio &lt; 6s),{' '}
                    <code>cinematic_slow_mo</code> (6–12s),{' '}
                    <code>loop_or_freeze</code> (&gt; 12s).
                  </div>
                  <div style={{ marginTop: '0.25rem', fontSize: '0.8rem', lineHeight: '1.5' }}>
                    📢 Uploaded clip audio is stripped — only TTS voiceover plays in the final video.
                  </div>
                </div>
              )}

              {videoVersion === 'v6' && (
                <div className="v6-info-banner" style={{
                  background: 'linear-gradient(135deg, rgba(168,85,247,0.12), rgba(236,72,153,0.08))',
                  border: '1px solid rgba(168,85,247,0.3)',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1rem',
                  marginBottom: '1rem',
                  fontSize: '0.85rem',
                  color: darkMode ? '#d8b4fe' : '#7c3aed',
                }}>
                  <strong>✨ V6 Hybrid Mode:</strong> Mix AI-generated image scenes and video clip scenes in one video.
                  <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', lineHeight: '1.5' }}>
                    <strong>Image scenes</strong> — AI generates the image; you choose a zoom effect and optionally click a focus point.
                    Effects: <code>none</code> (static), <code>zoom_in</code> (default), <code>zoom_out</code>, <code>ken_burns</code>.
                  </div>
                  <div style={{ marginTop: '0.25rem', fontSize: '0.8rem', lineHeight: '1.5' }}>
                    <strong>Video scenes</strong> — Upload a clip (e.g. from Grok); it is automatically time-fitted to match the voiceover.
                    Strategies: <code>auto</code>, <code>trim</code>, <code>cinematic_slow_mo</code>, <code>loop_or_freeze</code>.
                  </div>
                  <div style={{ marginTop: '0.4rem', fontSize: '0.8rem', lineHeight: '1.5', borderTop: '1px solid rgba(168,85,247,0.2)', paddingTop: '0.4rem' }}>
                    🎯 <strong>After asset generation</strong> you can set a custom zoom focus point on each image scene by clicking on the image.
                  </div>
                </div>
              )}

              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <label style={{ margin: 0 }}>Paste your scenes JSON array below</label>
                  {scenesText && (
                    <span className={`scene-count ${sceneInfo.valid ? 'valid' : 'invalid'}`}>
                      {sceneInfo.valid ? `✓ ${sceneInfo.count} ${(videoVersion === 'v2' || videoVersion === 'v3') ? 'grouped scene' : 'scene'}${sceneInfo.count !== 1 ? 's' : ''}` : `✗ ${sceneInfo.error}`}
                    </span>
                  )}
                </div>
                <textarea
                  className="scene-editor"
                  placeholder={
                    videoVersion === 'v6' ? DEFAULT_V6_SCENES_EXAMPLE :
                    videoVersion === 'v5' ? DEFAULT_V5_SCENES_EXAMPLE :
                    (videoVersion === 'v2' || videoVersion === 'v3') ? DEFAULT_V2_SCENES_EXAMPLE :
                    DEFAULT_SCENES_EXAMPLE
                  }
                  value={scenesText}
                  onChange={e => setScenesText(e.target.value)}
                />
                <p className="help-text">
                  {videoVersion === 'v6'
                    ? 'Each V6 scene needs "voiceover", "prompt", and "media_type" ("image" or "video"). Image scenes support "zoom_effect" and "focus_x"/"focus_y". Video scenes support "time_fit_strategy".'
                    : videoVersion === 'v5'
                    ? 'Each V5 scene needs "voiceover", "prompt", and optionally "media_type" and "time_fit_strategy".'
                    : (videoVersion === 'v2' || videoVersion === 'v3')
                    ? `Each ${videoVersion.toUpperCase()} scene needs "voiceover", "prompts" (array), and "visual_beats" (array with trigger_word, effect, image_index).`
                    : 'Each scene needs a "voiceover" (text for speech) and a "prompt" (text for image generation).'}
                </p>
              </div>
            </div>

            {/* ─── Workflow Stepper ─── */}
            <div className="card">
              <div className="card-header">
                <h2>🔄 Workflow</h2>
                <span className="card-badge">Steps</span>
              </div>

              {/* Step Indicator */}
              <div className="workflow-stepper">
                <div className={`workflow-step ${workflowStep >= 1 ? 'active' : ''} ${workflowStep > 1 ? 'completed' : ''}`}>
                  <div className="step-number">{workflowStep > 1 ? '✓' : '1'}</div>
                  <div className="step-label">{(videoVersion === 'v5' || videoVersion === 'v6') ? 'Generate Assets' : 'Generate Audio & Images'}</div>
                </div>
                <div className={`step-connector ${workflowStep > 1 ? 'active' : ''}`}>
                  <span className="step-arrow">→</span>
                </div>
                <div className={`workflow-step ${workflowStep >= 2 ? 'active' : ''} ${workflowStep > 2 ? 'completed' : ''}`}>
                  <div className="step-number">{workflowStep > 2 ? '✓' : '2'}</div>
                  <div className="step-label">{videoVersion === 'v5' ? 'Upload Video Clips' : videoVersion === 'v6' ? 'Review & Configure' : 'Review Resources'}</div>
                </div>
                <div className={`step-connector ${workflowStep > 2 ? 'active' : ''}`}>
                  <span className="step-arrow">→</span>
                </div>
                <div className={`workflow-step ${workflowStep >= 3 ? 'active' : ''} ${workflowStep > 3 ? 'completed' : ''}`}>
                  <div className="step-number">{workflowStep > 3 ? '✓' : '3'}</div>
                  <div className="step-label">Prepare Video</div>
                </div>
              </div>

              {/* Step 1: Generate Audio & Images */}
              {workflowStep === 0 && (
                <div className="workflow-action">
                  <button
                    className="btn btn-primary btn-block btn-glow"
                    disabled={!sceneInfo.valid || isGenerating}
                    onClick={startGeneration}
                  >
                    {videoVersion === 'v5' ? '🎙️ Generate Voiceover Audio' : videoVersion === 'v6' ? '✨ Generate V6 Assets' : '🎙️🖼️ Generate Audio & Images'}
                  </button>
                  <p className="help-text" style={{ textAlign: 'center', marginTop: '0.5rem' }}>
                    {videoVersion === 'v5'
                      ? 'This will generate TTS voiceover audio for all scenes. You will then upload video clips for each scene.'
                      : videoVersion === 'v6'
                      ? 'This will generate TTS audio for all scenes and AI images for image scenes. Video scenes will need clip uploads.'
                      : 'This will generate voiceover audio and images for all scenes. You can review them before creating the video.'}
                  </p>
                </div>
              )}

              {/* Step 1 In Progress */}
              {workflowStep === 1 && (
                <div className="workflow-action">
                  <div className="progress-bar-wrapper">
                    <div className={`progress-bar-fill ${jobStatus}`} style={{ width: `${jobProgress}%` }}></div>
                  </div>
                  <div className="progress-info">
                    <span className="progress-message">{jobMessage}</span>
                    <span className={`progress-percent ${jobStatus}`}>{jobProgress}%</span>
                  </div>
                  {jobStatus === 'failed' && (
                    <div className="retry-area" style={{ marginTop: '1rem' }}>
                      <span style={{ fontSize: '1.5rem' }}>🔄</span>
                      <div>
                        <strong>Generation Failed</strong>
                        <p className="muted">Click retry to resume from where it stopped.</p>
                      </div>
                      <button className="btn btn-primary" onClick={retryGeneration} disabled={isGenerating}>
                        {isGenerating ? <><span className="spinner"></span> Retrying...</> : <>🔄 Retry</>}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Step 2: Review Resources */}
              {workflowStep === 2 && (
                <div className="workflow-action">
                  <div className="review-header">
                    <h3>{videoVersion === 'v6' ? '✨ V6 Asset Dashboard' : videoVersion === 'v5' ? '🎬 V5 Asset Dashboard' : '📋 Review Generated Resources'}</h3>
                    <p className="muted">
                      {videoVersion === 'v6'
                        ? 'Review images and set zoom effects for image scenes. Upload video clips for video scenes.'
                        : videoVersion === 'v5'
                        ? 'Copy each prompt, generate a 6-second video clip in Grok, then upload the .mp4 file for each scene.'
                        : 'Check each image below. Click "Regenerate" on any image that doesn\'t look right.'}
                    </p>
                  </div>

                  {videoVersion === 'v6' ? (
                    /* V6 Asset Dashboard: image scenes + video scenes */
                    <div className="review-grid" style={{ gridTemplateColumns: '1fr' }}>
                      {reviewAssets.map((asset, idx) => {
                        const isImageScene = asset.media_type !== 'video'
                        const isVideoReady = !isImageScene && (asset.has_video || v6UploadedVideos[idx])
                        const isImageReady = isImageScene && asset.has_image
                        const isReady = isImageScene ? isImageReady : isVideoReady
                        const currentFocus = v6DirectorFocus[idx] || { x: asset.focus_x || 0.5, y: asset.focus_y || 0.5 }
                        const currentZoom = v6DirectorZoom[idx] || asset.zoom_effect || 'zoom_in'
                        const isEditingFocus = v6DirectorSceneIndex === idx

                        return (
                          <div key={idx} className="review-card" style={{
                            gridColumn: '1 / -1',
                            border: isReady ? '2px solid rgba(168,85,247,0.45)' : '2px solid rgba(245,158,11,0.3)',
                            background: isReady ? 'rgba(168,85,247,0.05)' : 'rgba(245,158,11,0.03)',
                          }}>
                            <div className="review-card-header" style={{ marginBottom: '0.75rem' }}>
                              <span className="review-scene-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                {isReady ? '✅' : '⏳'} Scene {idx + 1}
                                <code style={{ fontSize: '0.7rem', opacity: 0.7, padding: '0.1rem 0.4rem', background: isImageScene ? 'rgba(168,85,247,0.15)' : 'rgba(16,185,129,0.15)', borderRadius: '0.25rem' }}>
                                  {isImageScene ? '🖼 image' : '🎬 video'}
                                </code>
                                {isImageScene && (
                                  <code style={{ fontSize: '0.7rem', opacity: 0.7, padding: '0.1rem 0.4rem', background: 'rgba(99,102,241,0.15)', borderRadius: '0.25rem' }}>
                                    {currentZoom}
                                  </code>
                                )}
                              </span>
                            </div>

                            {/* Voiceover */}
                            <div style={{ marginBottom: '0.75rem' }}>
                              <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6, marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Voiceover</div>
                              <p style={{ fontSize: '0.9rem', lineHeight: '1.6', margin: 0, fontStyle: 'italic' }}>
                                &ldquo;{asset.voiceover}&rdquo;
                              </p>
                            </div>

                            {/* Audio */}
                            {asset.has_audio && (
                              <audio controls src={asset.audio_url} style={{ width: '100%', marginBottom: '0.75rem' }} />
                            )}

                            {/* Prompt */}
                            <div style={{ background: darkMode ? 'rgba(0,0,0,0.25)' : 'rgba(0,0,0,0.05)', borderRadius: '0.5rem', padding: '0.75rem', marginBottom: '0.75rem' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.75rem' }}>
                                <div style={{ flex: 1 }}>
                                  <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6, marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Visual Prompt</div>
                                  <p style={{ fontSize: '0.85rem', lineHeight: '1.5', margin: 0 }}>{asset.prompt}</p>
                                </div>
                                {!isImageScene && (
                                  <button className="btn btn-secondary btn-sm" onClick={() => v6CopyPrompt(asset.prompt, idx)} style={{ whiteSpace: 'nowrap', flexShrink: 0 }}>
                                    {v6CopiedIndex === idx ? '✅ Copied!' : '📋 Copy Prompt'}
                                  </button>
                                )}
                              </div>
                            </div>

                            {/* Image scene: show image + zoom controls */}
                            {isImageScene && (
                              <div>
                                {asset.has_image ? (
                                  <div>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                      <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Generated Image</div>
                                      <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                                        <select
                                          value={currentZoom}
                                          onChange={e => v6SaveFocusPoint(idx, currentFocus.x, currentFocus.y, e.target.value)}
                                          style={{ fontSize: '0.75rem', padding: '0.2rem 0.4rem', borderRadius: '0.3rem', border: '1px solid rgba(168,85,247,0.3)', background: darkMode ? '#1e1e2e' : '#fff', color: 'inherit' }}
                                        >
                                          <option value="none">🖼 Static (none)</option>
                                          <option value="zoom_in">🔍 Zoom In</option>
                                          <option value="zoom_out">🔎 Zoom Out</option>
                                          <option value="ken_burns">🎥 Ken Burns</option>
                                        </select>
                                        <button
                                          className={`btn btn-sm ${isEditingFocus ? 'btn-primary' : 'btn-secondary'}`}
                                          onClick={() => setV6DirectorSceneIndex(isEditingFocus ? null : idx)}
                                          title="Click to set zoom focus point on image"
                                          style={{ fontSize: '0.75rem' }}
                                        >
                                          {isEditingFocus ? '✅ Done' : '🎯 Set Focus'}
                                        </button>
                                        <button
                                          className="btn btn-secondary btn-sm"
                                          disabled={v6RegeneratingIndex === idx}
                                          onClick={() => v6RegenerateImage(idx)}
                                          style={{ fontSize: '0.75rem' }}
                                        >
                                          {v6RegeneratingIndex === idx ? <><span className="spinner"></span></> : '🔄'}
                                        </button>
                                      </div>
                                    </div>
                                    <div
                                      style={{ position: 'relative', cursor: isEditingFocus ? 'crosshair' : 'default', borderRadius: '0.5rem', overflow: 'hidden', border: isEditingFocus ? '2px solid rgba(168,85,247,0.6)' : '1px solid rgba(168,85,247,0.2)', display: 'inline-block', maxWidth: '100%', width: '100%' }}
                                      onClick={isEditingFocus ? (e) => {
                                        const rect = e.currentTarget.getBoundingClientRect()
                                        const x = Math.round(((e.clientX - rect.left) / rect.width) * 100) / 100
                                        const y = Math.round(((e.clientY - rect.top) / rect.height) * 100) / 100
                                        v6SaveFocusPoint(idx, x, y, currentZoom)
                                      } : undefined}
                                    >
                                      <img src={asset.image_url} alt={`V6 Scene ${idx + 1}`} style={{ width: '100%', display: 'block' }} />
                                      {isEditingFocus && (
                                        <div style={{
                                          position: 'absolute',
                                          left: `${currentFocus.x * 100}%`,
                                          top: `${currentFocus.y * 100}%`,
                                          transform: 'translate(-50%, -50%)',
                                          width: '20px',
                                          height: '20px',
                                          borderRadius: '50%',
                                          border: '3px solid #a855f7',
                                          background: 'rgba(168,85,247,0.4)',
                                          pointerEvents: 'none',
                                          boxShadow: '0 0 0 2px #fff',
                                        }} />
                                      )}
                                    </div>
                                    {isEditingFocus && (
                                      <p style={{ fontSize: '0.75rem', opacity: 0.65, marginTop: '0.4rem', textAlign: 'center' }}>
                                        Click on the image to set the zoom focus point. Current: ({Math.round(currentFocus.x * 100)}%, {Math.round(currentFocus.y * 100)}%)
                                      </p>
                                    )}
                                  </div>
                                ) : (
                                  <div style={{ padding: '1rem', textAlign: 'center', opacity: 0.6, fontSize: '0.85rem' }}>⏳ Image generating…</div>
                                )}
                              </div>
                            )}

                            {/* Video scene: upload area */}
                            {!isImageScene && (
                              <div>
                                {(asset.has_video || v6UploadedVideos[idx]) ? (
                                  <div>
                                    <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6, marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Uploaded Video</div>
                                    <video controls src={asset.video_url} style={{ width: '100%', maxHeight: '300px', borderRadius: '0.5rem', background: '#000' }} />
                                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                                      <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                                        🔄 Replace
                                        <input type="file" accept=".mp4,video/mp4" style={{ display: 'none' }} onChange={e => { if (e.target.files?.[0]) v6UploadVideo(idx, e.target.files[0]) }} />
                                      </label>
                                    </div>
                                  </div>
                                ) : (
                                  <div style={{ border: '2px dashed rgba(245,158,11,0.3)', borderRadius: '0.75rem', padding: '1.5rem', textAlign: 'center', background: darkMode ? 'rgba(0,0,0,0.15)' : 'rgba(0,0,0,0.03)' }}>
                                    <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🎥</div>
                                    <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.85rem', opacity: 0.7 }}>
                                      Generate a clip using the prompt above, then upload the .mp4 file
                                    </p>
                                    <label className="btn btn-primary" style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                                      {v6Uploading === idx ? <><span className="spinner"></span> Uploading...</> : <>📤 Upload Video (.mp4)</>}
                                      <input type="file" accept=".mp4,video/mp4" style={{ display: 'none' }} disabled={v6Uploading === idx} onChange={e => { if (e.target.files?.[0]) v6UploadVideo(idx, e.target.files[0]) }} />
                                    </label>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  ) : videoVersion === 'v5' ? (
                    /* V5 Asset Dashboard: upload video clips for each scene */
                    <div className="review-grid" style={{ gridTemplateColumns: '1fr' }}>
                      {reviewAssets.map((asset, idx) => (
                        <div key={idx} className="review-card" style={{
                          gridColumn: '1 / -1',
                          border: (asset.has_video || v5UploadedVideos[idx])
                            ? '2px solid rgba(16,185,129,0.5)'
                            : '2px solid rgba(245,158,11,0.3)',
                          background: (asset.has_video || v5UploadedVideos[idx])
                            ? 'rgba(16,185,129,0.05)'
                            : 'rgba(245,158,11,0.03)',
                        }}>
                          <div className="review-card-header" style={{ marginBottom: '0.75rem' }}>
                            <span className="review-scene-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              {(asset.has_video || v5UploadedVideos[idx]) ? '✅' : '⏳'} Scene {idx + 1}
                              {asset.time_fit_strategy && (
                                <code style={{ fontSize: '0.7rem', opacity: 0.7, padding: '0.1rem 0.4rem', background: 'rgba(99,102,241,0.15)', borderRadius: '0.25rem' }}>
                                  {asset.time_fit_strategy}
                                </code>
                              )}
                            </span>
                          </div>

                          {/* Voiceover text */}
                          <div style={{ marginBottom: '0.75rem' }}>
                            <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6, marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Voiceover</div>
                            <p style={{ fontSize: '0.9rem', lineHeight: '1.6', margin: 0, fontStyle: 'italic' }}>
                              &ldquo;{asset.voiceover}&rdquo;
                            </p>
                          </div>

                          {/* Audio player */}
                          {asset.has_audio && (
                            <audio controls src={asset.audio_url} style={{ width: '100%', marginBottom: '0.75rem' }} />
                          )}

                          {/* Visual prompt + copy button */}
                          <div style={{
                            background: darkMode ? 'rgba(0,0,0,0.25)' : 'rgba(0,0,0,0.05)',
                            borderRadius: '0.5rem',
                            padding: '0.75rem',
                            marginBottom: '0.75rem',
                          }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.75rem' }}>
                              <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6, marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Visual Prompt</div>
                                <p style={{ fontSize: '0.85rem', lineHeight: '1.5', margin: 0 }}>{asset.prompt}</p>
                              </div>
                              <button
                                className="btn btn-secondary btn-sm"
                                onClick={() => v5CopyPrompt(asset.prompt, idx)}
                                style={{ whiteSpace: 'nowrap', flexShrink: 0 }}
                              >
                                {v5CopiedIndex === idx ? '✅ Copied!' : '📋 Copy Prompt'}
                              </button>
                            </div>
                          </div>

                          {/* Upload / Preview area */}
                          {(asset.has_video || v5UploadedVideos[idx]) ? (
                            <div>
                              <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6, marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Uploaded Video</div>
                              <video
                                controls
                                src={asset.video_url}
                                style={{ width: '100%', maxHeight: '300px', borderRadius: '0.5rem', background: '#000' }}
                              />
                              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                                <label
                                  className="btn btn-secondary btn-sm"
                                  style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
                                >
                                  🔄 Replace
                                  <input
                                    type="file"
                                    accept=".mp4,video/mp4"
                                    style={{ display: 'none' }}
                                    onChange={e => { if (e.target.files?.[0]) v5UploadVideo(idx, e.target.files[0]) }}
                                  />
                                </label>
                              </div>
                            </div>
                          ) : (
                            <div style={{
                              border: '2px dashed rgba(245,158,11,0.3)',
                              borderRadius: '0.75rem',
                              padding: '1.5rem',
                              textAlign: 'center',
                              background: darkMode ? 'rgba(0,0,0,0.15)' : 'rgba(0,0,0,0.03)',
                            }}>
                              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🎥</div>
                              <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.85rem', opacity: 0.7 }}>
                                Generate a 6-second clip using the prompt above, then upload the .mp4 file
                              </p>
                              <label
                                className="btn btn-primary"
                                style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                              >
                                {v5Uploading === idx ? (
                                  <><span className="spinner"></span> Uploading...</>
                                ) : (
                                  <>📤 Upload Video (.mp4)</>
                                )}
                                <input
                                  type="file"
                                  accept=".mp4,video/mp4"
                                  style={{ display: 'none' }}
                                  disabled={v5Uploading === idx}
                                  onChange={e => { if (e.target.files?.[0]) v5UploadVideo(idx, e.target.files[0]) }}
                                />
                              </label>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (videoVersion === 'v2' || videoVersion === 'v3') ? (
                    /* V2 Review: grouped scenes with multiple images */
                    <div className="review-grid">
                      {reviewAssets.map((scene, sIdx) => (
                        <div key={sIdx} className="review-card" style={{ gridColumn: '1 / -1' }}>
                          <div className="review-card-header">
                            <span className="review-scene-label">Grouped Scene {sIdx + 1}</span>
                          </div>
                          <p className="review-prompt" style={{ marginBottom: '0.5rem', fontStyle: 'italic' }}>
                            {scene.voiceover && scene.voiceover.length > 150
                              ? scene.voiceover.substring(0, 150) + '...'
                              : scene.voiceover}
                          </p>
                          {scene.has_audio && (
                            <audio controls src={scene.audio_url} style={{ width: '100%', marginBottom: '0.75rem' }} />
                          )}
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.75rem' }}>
                            {(scene.images || []).map((img, pIdx) => (
                              <div key={pIdx} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: '0.5rem', padding: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                                  <span style={{ fontSize: '0.75rem', opacity: 0.7 }}>Image {pIdx + 1}</span>
                                  <button
                                    className="btn btn-secondary btn-sm"
                                    disabled={regeneratingIndex === `${sIdx}_${pIdx}`}
                                    onClick={() => regenerateV2Image(sIdx, pIdx)}
                                    style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem' }}
                                  >
                                    {regeneratingIndex === `${sIdx}_${pIdx}` ? (
                                      <><span className="spinner"></span> ...</>
                                    ) : (
                                      <>🔄</>
                                    )}
                                  </button>
                                </div>
                                <div className="review-image-wrapper">
                                  {img.has_image ? (
                                    <img src={img.image_url} alt={`Scene ${sIdx + 1} Image ${pIdx + 1}`} className="review-image" />
                                  ) : (
                                    <div className="review-image-placeholder">No image</div>
                                  )}
                                </div>
                                <p className="review-prompt" title={img.prompt} style={{ fontSize: '0.75rem', marginTop: '0.3rem' }}>
                                  {img.prompt && img.prompt.length > 60 ? img.prompt.substring(0, 60) + '...' : img.prompt}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    /* V1 Review: one image per scene */
                    <div className="review-grid">
                      {reviewAssets.map((asset, i) => (
                        <div key={i} className="review-card">
                          <div className="review-card-header">
                            <span className="review-scene-label">Scene {i + 1}</span>
                            <button
                              className="btn btn-secondary btn-sm"
                              disabled={regeneratingIndex === i}
                              onClick={() => regenerateImage(i)}
                            >
                              {regeneratingIndex === i ? (
                                <><span className="spinner"></span> Regenerating...</>
                              ) : (
                                <>🔄 Regenerate</>
                              )}
                            </button>
                          </div>
                          <div className="review-image-wrapper">
                            {asset.has_image ? (
                              <img src={asset.image_url} alt={`Scene ${i + 1}`} className="review-image" />
                            ) : (
                              <div className="review-image-placeholder">No image generated</div>
                            )}
                          </div>
                          <p className="review-prompt" title={asset.prompt}>
                            {asset.prompt.length > 100 ? asset.prompt.substring(0, 100) + '...' : asset.prompt}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {videoVersion === 'v6' ? (
                    /* V6: Approve button (video scenes must all be uploaded) */
                    <div className="review-actions">
                      {!v6AllVideoScenesUploaded() ? (
                        <div style={{
                          padding: '0.75rem 1rem',
                          background: 'rgba(245,158,11,0.1)',
                          border: '1px solid rgba(245,158,11,0.25)',
                          borderRadius: '0.5rem',
                          textAlign: 'center',
                          fontSize: '0.9rem',
                        }}>
                          ⏳ Upload video clips for all video scenes to proceed
                          <span style={{ marginLeft: '0.5rem', fontWeight: 600 }}>
                            ({reviewAssets.filter((a, i) => a.media_type !== 'video' || a.has_video || v6UploadedVideos[i]).length}/{reviewAssets.length})
                          </span>
                        </div>
                      ) : (
                        <div>
                          <button
                            className="btn btn-success btn-block btn-glow"
                            onClick={() => { approveAssets() }}
                            disabled={assetsApproved}
                          >
                            {assetsApproved ? '✅ All Scenes Ready' : '✅ Confirm All Scenes & Continue'}
                          </button>
                        </div>
                      )}
                    </div>
                  ) : videoVersion === 'v5' ? (
                    /* V5: Show progress indicator + prepare button when all videos uploaded */
                    <div className="review-actions">
                      {!v5AllVideosUploaded() ? (
                        <div style={{
                          padding: '0.75rem 1rem',
                          background: 'rgba(245,158,11,0.1)',
                          border: '1px solid rgba(245,158,11,0.25)',
                          borderRadius: '0.5rem',
                          textAlign: 'center',
                          fontSize: '0.9rem',
                        }}>
                          ⏳ Upload video clips for all {reviewAssets.length} scene{reviewAssets.length !== 1 ? 's' : ''} to proceed
                          <span style={{ marginLeft: '0.5rem', fontWeight: 600 }}>
                            ({Object.keys(v5UploadedVideos).length + reviewAssets.filter(a => a.has_video).length}/{reviewAssets.length})
                          </span>
                        </div>
                      ) : (
                        <div>
                          <button
                            className="btn btn-success btn-block btn-glow"
                            onClick={() => { approveAssets() }}
                            disabled={assetsApproved}
                          >
                            {assetsApproved ? '✅ All Clips Ready' : '✅ Confirm All Clips & Continue'}
                          </button>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="review-actions">
                      <button
                        className="btn btn-success btn-block btn-glow"
                        onClick={() => { approveAssets(); if (videoVersion === 'v3') initDirectorReview(); }}
                        disabled={assetsApproved}
                      >
                        {assetsApproved ? '✅ Approved' : '✅ Approve & Continue'}
                      </button>
                    </div>
                  )}

                  {/* V3 Director Review UI */}
                  {assetsApproved && videoVersion === 'v3' && !directorReviewDone && directorReviewBeats.length > 0 && (
                    <div className="director-review" style={{
                      marginTop: '1.5rem',
                      background: 'linear-gradient(135deg, rgba(245,158,11,0.08), rgba(239,68,68,0.06))',
                      border: '1px solid rgba(245,158,11,0.3)',
                      borderRadius: '0.75rem',
                      padding: '1.25rem',
                    }}>
                      <h3 style={{ margin: '0 0 0.25rem 0', fontSize: '1.1rem' }}>
                        🎯 Action Required: Select Zoom Focus Point for &quot;{directorReviewBeats[directorReviewIndex]?.triggerWord}&quot;
                      </h3>
                      <p className="muted" style={{ margin: '0 0 0.75rem 0', fontSize: '0.85rem' }}>
                        Beat {directorReviewIndex + 1} of {directorReviewBeats.length} &middot; Effect: <code>{directorReviewBeats[directorReviewIndex]?.effect}</code>
                      </p>

                      <div
                        style={{
                          position: 'relative',
                          cursor: 'crosshair',
                          borderRadius: '0.5rem',
                          overflow: 'hidden',
                          border: '2px solid rgba(245,158,11,0.4)',
                          display: 'inline-block',
                          maxWidth: '100%',
                        }}
                        onClick={handleDirectorImageClick}
                      >
                        <img
                          src={directorReviewBeats[directorReviewIndex]?.imageUrl}
                          alt="Focus target"
                          style={{ display: 'block', maxWidth: '100%', maxHeight: '450px', objectFit: 'contain' }}
                          draggable={false}
                        />
                        {directorFocus && (
                          <div style={{
                            position: 'absolute',
                            left: `${directorFocus.x * 100}%`,
                            top: `${directorFocus.y * 100}%`,
                            transform: 'translate(-50%, -50%)',
                            pointerEvents: 'none',
                          }}>
                            {/* Red target icon */}
                            <div style={{
                              width: '28px', height: '28px',
                              border: '3px solid #ef4444',
                              borderRadius: '50%',
                              position: 'relative',
                              boxShadow: '0 0 0 2px rgba(0,0,0,0.3), inset 0 0 0 2px rgba(0,0,0,0.15)',
                            }}>
                              <div style={{
                                position: 'absolute', top: '50%', left: '50%',
                                transform: 'translate(-50%, -50%)',
                                width: '6px', height: '6px',
                                background: '#ef4444',
                                borderRadius: '50%',
                              }} />
                              {/* Crosshair lines */}
                              <div style={{ position: 'absolute', top: '-8px', left: '50%', transform: 'translateX(-50%)', width: '2px', height: '8px', background: '#ef4444' }} />
                              <div style={{ position: 'absolute', bottom: '-8px', left: '50%', transform: 'translateX(-50%)', width: '2px', height: '8px', background: '#ef4444' }} />
                              <div style={{ position: 'absolute', left: '-8px', top: '50%', transform: 'translateY(-50%)', width: '8px', height: '2px', background: '#ef4444' }} />
                              <div style={{ position: 'absolute', right: '-8px', top: '50%', transform: 'translateY(-50%)', width: '8px', height: '2px', background: '#ef4444' }} />
                            </div>
                          </div>
                        )}
                      </div>

                      {directorFocus && (
                        <p style={{ fontSize: '0.8rem', marginTop: '0.5rem', opacity: 0.7 }}>
                          Focus: ({directorFocus.x.toFixed(3)}, {directorFocus.y.toFixed(3)}) — click again to reposition
                        </p>
                      )}

                      <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
                        <button className="btn btn-primary" onClick={acceptFocusPoint}>
                          ✅ Accept
                        </button>
                        <button className="btn btn-secondary" onClick={skipFocusPoint}>
                          ⏭️ Skip (Use Center)
                        </button>
                      </div>
                    </div>
                  )}

                  {/* For V2: straight to Prepare Video after approval */}
                  {/* For V3: show Prepare Video only after director review is complete */}
                  {assetsApproved && (videoVersion !== 'v3' || directorReviewDone) && (
                    <div className="next-step-hint">
                      <span className="next-step-arrow">↓</span>
                      <button
                        className="btn btn-primary btn-block btn-glow"
                        onClick={prepareVideo}
                        disabled={isGenerating}
                      >
                        {isGenerating ? (
                          <><span className="spinner"></span> Preparing...</>
                        ) : (
                          <>🎬 Prepare Video</>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Step 3: Video Preparation In Progress */}
              {workflowStep === 3 && (
                <div className="workflow-action">
                  <div className="progress-bar-wrapper">
                    <div className={`progress-bar-fill ${jobStatus}`} style={{ width: `${jobProgress}%` }}></div>
                  </div>
                  <div className="progress-info">
                    <span className="progress-message">{jobMessage}</span>
                    <span className={`progress-percent ${jobStatus}`}>{jobProgress}%</span>
                  </div>
                  {jobStatus === 'failed' && (
                    <div className="retry-area" style={{ marginTop: '1rem' }}>
                      <span style={{ fontSize: '1.5rem' }}>🔄</span>
                      <div>
                        <strong>Video Preparation Failed</strong>
                        <p className="muted">Click retry to try assembling the video again.</p>
                      </div>
                      <button className="btn btn-primary" onClick={prepareVideo} disabled={isGenerating}>
                        {isGenerating ? <><span className="spinner"></span> Retrying...</> : <>🔄 Retry</>}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Step 4: Completed */}
              {workflowStep === 4 && jobStatus === 'completed' && (
                <div className="workflow-action">
                  <div className="download-area">
                    <span style={{ fontSize: '2rem' }}>🎉</span>
                    <div>
                      <strong>Video Ready!</strong>
                      <p className="muted">Your video has been generated successfully.</p>
                    </div>
                    <button className="btn btn-success" onClick={downloadVideo}>
                      ⬇️ Download Video
                    </button>
                  </div>

                  <div className="preview-section">
                    <div className="preview-header">
                      <h3>🎬 Preview</h3>
                      <p className="muted">Watch your generated video right here</p>
                    </div>
                    <div className="preview-player">
                      <video
                        controls
                        src={`${API_BASE}/download/${jobId}`}
                        className="video-player"
                      />
                    </div>
                  </div>

                  {/* ─ Time-fit Strategy Editor (V5 / V6 video-clip scenes) ─ */}
                  {(videoVersion === 'v5' || videoVersion === 'v6') && reviewAssets.length > 0 && (
                    <div style={{
                      marginTop: '1.5rem',
                      background: 'linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.06))',
                      border: '1px solid rgba(99,102,241,0.3)',
                      borderRadius: '0.75rem',
                      padding: '1.25rem',
                    }}>
                      <h3 style={{ margin: '0 0 0.25rem 0', fontSize: '1.05rem' }}>⏱️ Adjust Time-Fit &amp; Re-render</h3>
                      <p className="muted" style={{ margin: '0 0 1rem 0', fontSize: '0.85rem' }}>
                        Change how video clips are fitted to the voiceover length, then re-render without regenerating audio or images.
                      </p>
                      {reviewAssets.map((asset, idx) => {
                        const isVideoScene = videoVersion === 'v5' || asset.media_type === 'video'
                        if (!isVideoScene) return null
                        const currentStrategy = timeFitEdits[idx] !== undefined
                          ? timeFitEdits[idx]
                          : (asset.time_fit_strategy || 'auto')
                        return (
                          <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.6rem', flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '0.85rem', minWidth: '5rem' }}>
                              Scene {idx + 1}
                            </span>
                            <select
                              value={currentStrategy}
                              onChange={e => setTimeFitEdits(prev => ({ ...prev, [idx]: e.target.value }))}
                              style={{ flex: 1, minWidth: '12rem', maxWidth: '18rem' }}
                            >
                              <option value="auto">Auto (smart fit)</option>
                              <option value="trim">Trim to voiceover</option>
                              <option value="cinematic_slow_mo">Cinematic Slow-Mo</option>
                              <option value="loop_or_freeze">Loop / Freeze</option>
                            </select>
                          </div>
                        )
                      })}
                      <button
                        className="btn btn-primary"
                        style={{ marginTop: '0.75rem' }}
                        disabled={timeFitSaving || isGenerating}
                        onClick={async () => {
                          if (!jobId) return
                          setTimeFitSaving(true)
                          setError('')
                          try {
                            // Push all edits to backend
                            for (const [idxStr, strategy] of Object.entries(timeFitEdits)) {
                              const idx = parseInt(idxStr, 10)
                              const res = await fetch(`${API_BASE}/update-time-fit/${jobId}`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ scene_index: idx, time_fit_strategy: strategy }),
                              })
                              if (!res.ok) {
                                const err = await res.json()
                                throw new Error(err.detail || 'Failed to update scene')
                              }
                            }
                            // Re-render
                            await prepareVideo()
                          } catch (err) {
                            setError(err.message)
                          } finally {
                            setTimeFitSaving(false)
                          }
                        }}
                      >
                        {timeFitSaving ? <><span className="spinner"></span> Saving...</> : <>🔄 Re-render with New Settings</>}
                      </button>
                    </div>
                  )}

                  <button
                    className="btn btn-secondary btn-block"
                    style={{ marginTop: '1rem' }}
                    onClick={() => {
                      setWorkflowStep(0)
                      setJobId(null)
                      setJobStatus(null)
                      setJobProgress(0)
                      setJobMessage('')
                      setReviewAssets([])
                      setAssetsApproved(false)
                      setDirectorReviewBeats([])
                      setDirectorReviewIndex(0)
                      setDirectorFocus(null)
                      setDirectorReviewDone(false)
                      setV5UploadedVideos({})
                      setV5CopiedIndex(null)
                      setTimeFitEdits({})
                    }}
                  >
                    🔄 Start New Video
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── My Videos Page ─── */}
        {activeTab === 'myvideos' && (
          <div className="page-content">
            <div className="page-header">
              <h1>📼 My Videos</h1>
              <p className="page-subtitle">All previously generated videos. Click a job to download or re-render with updated settings.</p>
            </div>

            {myVideos.length === 0 ? (
              <div className="card" style={{ textAlign: 'center', padding: '2.5rem 1.5rem' }}>
                <div style={{ fontSize: '3rem', marginBottom: '0.75rem' }}>🎬</div>
                <p style={{ color: 'var(--text-muted)' }}>No videos generated yet. Head to the <strong>Create</strong> tab to get started.</p>
                <button className="btn btn-primary" style={{ marginTop: '1rem' }} onClick={() => setActiveTab('create')}>
                  ➕ Create First Video
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {myVideos.map((entry) => (
                  <div key={entry.jobId} className="card" style={{ display: 'flex', alignItems: 'flex-start', gap: '1.25rem', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '12rem' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.25rem' }}>
                        Job <code style={{ fontSize: '0.8rem', background: 'var(--bg-tertiary)', padding: '0.1rem 0.35rem', borderRadius: '4px' }}>{entry.jobId.slice(0, 8)}…</code>
                      </div>
                      <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                        Version: <strong>{entry.version?.toUpperCase() ?? 'V1'}</strong>
                        &nbsp;·&nbsp;
                        {new Date(entry.timestamp).toLocaleDateString()} {new Date(entry.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', alignItems: 'center' }}>
                      <a
                        href={`${API_BASE}/download/${entry.jobId}`}
                        download
                        className="btn btn-success"
                        style={{ textDecoration: 'none' }}
                      >
                        ⬇️ Download
                      </a>
                      <button
                        className="btn btn-secondary"
                        onClick={() => {
                          // Navigate to create tab and restore this job
                          setJobId(entry.jobId)
                          setVideoVersion(entry.version ?? 'v1')
                          setWorkflowStep(4)
                          setJobStatus('completed')
                          setActiveTab('create')
                        }}
                      >
                        🔍 View / Re-render
                      </button>
                      <button
                        className="btn btn-sm"
                        style={{ background: 'rgba(239,68,68,0.12)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)' }}
                        onClick={() => {
                          setMyVideos(prev => {
                            const updated = prev.filter(v => v.jobId !== entry.jobId)
                            try { localStorage.setItem('omniva_my_videos', JSON.stringify(updated)) } catch { /* ignore storage errors */ }
                            return updated
                          })
                        }}
                      >
                        🗑️ Remove
                      </button>
                    </div>
                  </div>
                ))}
                <button
                  className="btn btn-secondary"
                  style={{ alignSelf: 'flex-start', marginTop: '0.5rem' }}
                  onClick={() => {
                    if (window.confirm('Clear all history? This only removes the list entries — actual video files on the server are not deleted.')) {
                      setMyVideos([])
                      try { localStorage.removeItem('omniva_my_videos') } catch { /* ignore storage errors */ }
                    }
                  }}
                >
                  🗑️ Clear History
                </button>
              </div>
            )}
          </div>
        )}

        {/* ─── Test Lab Page ─── */}
        {activeTab === 'test' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Test Lab</h1>
              <p className="page-subtitle">Test individual audio and image generation before creating a full video</p>
            </div>

            {/* Test Audio Card — only audio-related settings */}
            <div className="card">
              <div className="card-header">
                <h2>🔊 Test Audio Generation</h2>
              </div>
              <p style={{ color: '#8b8fa3', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Test your speech settings with a sample text.
              </p>

              <div className="settings-section">
                <h3 className="section-title">Speech Settings</h3>
                {renderSpeechSettings()}
              </div>

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

            {/* Test Image Card — only image-related settings */}
            <div className="card">
              <div className="card-header">
                <h2>🖼️ Test Image Generation</h2>
              </div>
              <p style={{ color: '#8b8fa3', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Test your image settings with a sample prompt.
              </p>

              <div className="settings-section">
                <h3 className="section-title">Image Settings</h3>
                {renderImageSettings()}
              </div>

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
          </div>
        )}

        {/* ─── Templates Page (Placeholder) ─── */}
        {activeTab === 'templates' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Templates</h1>
              <p className="page-subtitle">Browse and use pre-built video templates to get started faster</p>
            </div>
            <div className="placeholder-grid">
              {[
                { icon: '🎓', title: 'Educational Explainer', desc: 'Create engaging educational videos with clear narration and illustrative images. Perfect for tutorials, courses, and how-to guides.' },
                { icon: '📢', title: 'Product Showcase', desc: 'Highlight product features with stunning visuals and professional voiceovers. Ideal for marketing and promotional content.' },
                { icon: '📖', title: 'Story Narration', desc: 'Bring stories to life with vivid imagery and expressive narration. Great for children\'s stories, podcasts, and audiobooks.' },
                { icon: '📰', title: 'News Recap', desc: 'Summarize news and current events with dynamic visuals and concise commentary. Perfect for social media and newsletters.' },
                { icon: '🏢', title: 'Corporate Presentation', desc: 'Build polished corporate presentations with data-driven visuals and professional tone. Ideal for stakeholder updates.' },
                { icon: '🌍', title: 'Travel Documentary', desc: 'Craft immersive travel documentaries with scenic imagery and engaging narration. Perfect for travel blogs and vlogs.' },
              ].map((tmpl, i) => (
                <div className="card template-card" key={i}>
                  <div className="template-icon">{tmpl.icon}</div>
                  <h3>{tmpl.title}</h3>
                  <p>{tmpl.desc}</p>
                  <span className="badge" style={{ marginTop: '0.75rem' }}>Coming Soon</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── Pricing Page ─── */}
        {activeTab === 'pricing' && (
          <div className="page-content">
            <div className="page-header" style={{ textAlign: 'center' }}>
              <h1>Pricing</h1>
              <p className="page-subtitle">Simple, transparent pricing for every creator</p>
            </div>

            {/* Africa-based crypto notice */}
            <div className="crypto-notice">
              <div className="crypto-notice-icon">🌍</div>
              <div className="crypto-notice-text">
                <strong>We are based in Africa</strong> — For now we only accept <strong>cryptocurrency</strong> payments. Credit card and PayPal support coming soon!
              </div>
            </div>

            <div className="pricing-grid">
              <div className="card pricing-card">
                <div className="pricing-tier">Free</div>
                <div className="pricing-price">$0<span className="pricing-period">/month</span></div>
                <ul className="pricing-features">
                  <li>✓ 5 videos per month</li>
                  <li>✓ Up to 720p resolution</li>
                  <li>✓ Landscape orientation</li>
                  <li>✓ Google Gemini provider</li>
                  <li>✓ Basic templates</li>
                  <li className="pricing-feature-disabled">✗ Portrait / vertical video</li>
                  <li className="pricing-feature-disabled">✗ Ken Burns effects</li>
                  <li className="pricing-feature-disabled">✗ Priority rendering</li>
                </ul>
                <button className="btn btn-secondary btn-block">Get Started Free</button>
              </div>
              <div className="card pricing-card pricing-card-featured">
                <div className="pricing-badge">Most Popular</div>
                <div className="pricing-tier">Pro</div>
                <div className="pricing-price">$19<span className="pricing-period">/month</span></div>
                <ul className="pricing-features">
                  <li>✓ 50 videos per month</li>
                  <li>✓ Up to 1080p resolution</li>
                  <li>✓ Landscape &amp; Portrait</li>
                  <li>✓ All AI providers</li>
                  <li>✓ All templates</li>
                  <li>✓ Ken Burns effects</li>
                  <li>✓ Custom models</li>
                  <li className="pricing-feature-disabled">✗ Priority rendering</li>
                </ul>
                <button className="btn btn-primary btn-block btn-glow" onClick={() => handleSelectPlan({ name: 'Pro', price: 19 })}>Pay with Crypto</button>
              </div>
              <div className="card pricing-card">
                <div className="pricing-tier">Enterprise</div>
                <div className="pricing-price">$49<span className="pricing-period">/month</span></div>
                <ul className="pricing-features">
                  <li>✓ Unlimited videos</li>
                  <li>✓ Up to 4K resolution</li>
                  <li>✓ All orientations</li>
                  <li>✓ All AI providers</li>
                  <li>✓ All templates</li>
                  <li>✓ All effects</li>
                  <li>✓ Custom models</li>
                  <li>✓ Priority rendering</li>
                </ul>
                <button className="btn btn-secondary btn-block" onClick={() => handleSelectPlan({ name: 'Enterprise', price: 49 })}>Pay with Crypto</button>
              </div>
            </div>

            {/* Payment methods info */}
            <div className="payment-methods-section">
              <h3>Accepted Payment Methods</h3>
              <div className="payment-methods-grid">
                <div className="payment-method-badge active">
                  <span className="payment-method-icon">₿</span>
                  <span>Bitcoin</span>
                  <span className="payment-method-status available">Available</span>
                </div>
                <div className="payment-method-badge active">
                  <span className="payment-method-icon">Ξ</span>
                  <span>Ethereum</span>
                  <span className="payment-method-status available">Available</span>
                </div>
                <div className="payment-method-badge active">
                  <span className="payment-method-icon">💲</span>
                  <span>USDC</span>
                  <span className="payment-method-status available">Available</span>
                </div>
                <div className="payment-method-badge active">
                  <span className="payment-method-icon">💵</span>
                  <span>USDT</span>
                  <span className="payment-method-status available">Available</span>
                </div>
                <div className="payment-method-badge disabled">
                  <span className="payment-method-icon">💳</span>
                  <span>Credit Card</span>
                  <span className="payment-method-status coming-soon">Coming Soon</span>
                </div>
                <div className="payment-method-badge disabled">
                  <span className="payment-method-icon">🅿️</span>
                  <span>PayPal</span>
                  <span className="payment-method-status coming-soon">Coming Soon</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── Crypto Payment Modal ─── */}
        {showPaymentModal && selectedPlan && (
          <div className="modal-overlay" onClick={() => setShowPaymentModal(false)}>
            <div className="modal-payment" onClick={e => e.stopPropagation()}>
              <button className="modal-close" onClick={() => setShowPaymentModal(false)}>✕</button>
              <div className="modal-payment-header">
                <h2>Pay with Crypto</h2>
                <p>Subscribe to <strong>{selectedPlan.name}</strong> — <strong>${selectedPlan.price}/month</strong></p>
              </div>

              {!selectedCrypto ? (
                <div className="crypto-select-section">
                  <h3>Choose your cryptocurrency</h3>
                  <div className="crypto-options-grid">
                    {cryptoOptions.map(crypto => (
                      <button
                        key={crypto.id}
                        className="crypto-option-card"
                        onClick={() => setSelectedCrypto(crypto)}
                      >
                        <span className="crypto-option-icon">{crypto.icon}</span>
                        <span className="crypto-option-name">{crypto.name}</span>
                        <span className="crypto-option-network">{crypto.network}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="crypto-payment-details">
                  <button className="crypto-back-btn" onClick={() => { setSelectedCrypto(null); setPaymentCopied(false) }}>
                    ← Back to options
                  </button>
                  <div className="crypto-selected-header">
                    <span className="crypto-selected-icon">{selectedCrypto.icon}</span>
                    <span className="crypto-selected-name">{selectedCrypto.name}</span>
                  </div>
                  <div className="crypto-amount-box">
                    <div className="crypto-amount-label">Amount to send</div>
                    <div className="crypto-amount-value">${selectedPlan.price}.00 USD</div>
                    <div className="crypto-amount-note">in {selectedCrypto.name} equivalent</div>
                  </div>
                  <div className="crypto-address-box">
                    <div className="crypto-address-label">Send to this address ({selectedCrypto.network})</div>
                    <div className="crypto-address-row">
                      <code className="crypto-address">{selectedCrypto.address}</code>
                      <button className="btn btn-sm" onClick={() => handleCopyAddress(selectedCrypto.address)}>
                        {paymentCopied ? '✓ Copied' : 'Copy'}
                      </button>
                    </div>
                  </div>
                  <div className="crypto-instructions">
                    <p><strong>Instructions:</strong></p>
                    <ol>
                      <li>Send the exact amount in {selectedCrypto.name} to the address above</li>
                      <li>After sending, save your transaction hash</li>
                      <li>Your plan will be activated within 30 minutes of confirmation</li>
                    </ol>
                  </div>
                  <div className="crypto-support-note">
                    Need help? Contact us at <strong>support@omnivalabs.com</strong>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── Settings Page ─── */}
        {activeTab === 'settings' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Settings</h1>
              <p className="page-subtitle">Manage your API keys and application preferences</p>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>🔑 API Keys</h2>
                <span className="card-badge">Required</span>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1.25rem', lineHeight: '1.6' }}>
                Add your API keys below to use the corresponding AI providers. Keys entered here take priority over server-configured keys.
                If left empty, the system will fall back to keys configured in the server&apos;s <code style={{ background: 'var(--bg-tertiary)', padding: '0.15rem 0.4rem', borderRadius: '4px', fontSize: '0.82rem' }}>.env</code> file.
              </p>

              {apiKeysSaved && (
                <div className="alert alert-success" style={{ marginBottom: '1rem' }}>
                  <span>✅ API keys saved to this session.</span>
                  <button onClick={() => setApiKeysSaved(false)} className="alert-close">✕</button>
                </div>
              )}

              <div className="api-key-grid">
                <div className="api-key-item">
                  <div className="api-key-header">
                    <span className="api-key-provider">✦ Google Gemini</span>
                    <span className={`api-key-status ${geminiApiKey ? 'configured' : ''}`}>
                      {geminiApiKey ? '● Configured' : '○ Not set'}
                    </span>
                  </div>
                  <p className="api-key-desc">Used for Gemini TTS (speech) and Gemini / Imagen image generation.</p>
                  <input
                    type="password"
                    placeholder="Enter your Gemini API key..."
                    value={geminiApiKey}
                    onChange={e => { setGeminiApiKey(e.target.value); setApiKeysSaved(false) }}
                  />
                </div>
                <div className="api-key-item">
                  <div className="api-key-header">
                    <span className="api-key-provider">◆ OpenAI</span>
                    <span className={`api-key-status ${openaiApiKey ? 'configured' : ''}`}>
                      {openaiApiKey ? '● Configured' : '○ Not set'}
                    </span>
                  </div>
                  <p className="api-key-desc">Used for OpenAI TTS voices and GPT Image / DALL·E image generation.</p>
                  <input
                    type="password"
                    placeholder="Enter your OpenAI API key..."
                    value={openaiApiKey}
                    onChange={e => { setOpenaiApiKey(e.target.value); setApiKeysSaved(false) }}
                  />
                </div>
                <div className="api-key-item">
                  <div className="api-key-header">
                    <span className="api-key-provider">▲ Together AI</span>
                    <span className={`api-key-status ${togetherApiKey ? 'configured' : ''}`}>
                      {togetherApiKey ? '● Configured' : '○ Not set'}
                    </span>
                  </div>
                  <p className="api-key-desc">Used for open-source image models like FLUX.1, Stable Diffusion XL, and DreamShaper.</p>
                  <input
                    type="password"
                    placeholder="Enter your Together AI API key..."
                    value={togetherApiKey}
                    onChange={e => { setTogetherApiKey(e.target.value); setApiKeysSaved(false) }}
                  />
                </div>
                <div className="api-key-item">
                  <div className="api-key-header">
                    <span className="api-key-provider">🎙️ ElevenLabs</span>
                    <span className={`api-key-status ${elevenlabsApiKey ? 'configured' : ''}`}>
                      {elevenlabsApiKey ? '● Configured' : '○ Not set'}
                    </span>
                  </div>
                  <p className="api-key-desc">Used for ElevenLabs TTS voices — ultra-realistic speech synthesis with multilingual and turbo models.</p>
                  <input
                    type="password"
                    placeholder="Enter your ElevenLabs API key..."
                    value={elevenlabsApiKey}
                    onChange={e => { setElevenlabsApiKey(e.target.value); setApiKeysSaved(false) }}
                  />
                </div>
              </div>
              <div style={{ marginTop: '1.25rem' }}>
                <button className="btn btn-primary" onClick={() => setApiKeysSaved(true)}>
                  💾 Save API Keys
                </button>
                <span style={{ marginLeft: '0.75rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Keys are stored in your browser session only — never sent to third parties.
                </span>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>🎨 Preferences</h2>
              </div>
              <div className="placeholder-content">
                <p>Customize your default generation settings, output formats, and notification preferences. Set default providers, models, and quality settings that will be applied to every new project automatically.</p>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>👤 Account</h2>
              </div>
              <div className="placeholder-content">
                <p>Manage your account details, subscription plan, and billing information. View your current plan limits and upgrade options for higher throughput and additional features.</p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-inner">
          <div className="footer-grid">
            <div className="footer-section">
              <h4 className="footer-brand">🎬 Omniva Video Forge</h4>
              <p>AI-powered video generation platform. Create stunning videos with multiple AI providers including Google Gemini, OpenAI, and Together AI.</p>
            </div>
            <div className="footer-section">
              <h4>Omniva Labs Products</h4>
              <ul className="footer-links">
                <li><span>🎬 Omniva Video Forge</span></li>
                <li><span>📈 Omniva Trading Journal</span></li>
                <li><span>🛒 Omniva Market Place</span></li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Resources</h4>
              <ul className="footer-links">
                <li><span>Documentation</span></li>
                <li><span>API Reference</span></li>
                <li><span>Templates</span></li>
                <li><span>Support</span></li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Company</h4>
              <ul className="footer-links">
                <li><span>About Omniva Labs</span></li>
                <li><span>Privacy Policy</span></li>
                <li><span>Terms of Service</span></li>
                <li><span>Contact</span></li>
              </ul>
            </div>
          </div>
          <div className="footer-bottom">
            <p>© {new Date().getFullYear()} Omniva Labs. All rights reserved. — Built by a solo developer with ❤️</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
