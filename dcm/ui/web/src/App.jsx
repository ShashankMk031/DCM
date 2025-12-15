import { useState, useEffect } from 'react'
import axios from 'axios'
import { Music, Settings, ListMusic, Plus, RefreshCw } from 'lucide-react'
import Player from './components/Player'
import RecommendationList from './components/RecommendationList'

// Configure Axios base URL
axios.defaults.baseURL = 'http://127.0.0.1:8000';

function App() {
  const [status, setStatus] = useState(null)
  const [recommendations, setRecommendations] = useState([])
  const [currentView, setCurrentView] = useState('recommendations')
  const [isLoading, setIsLoading] = useState(false)

  // Poll for player status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await axios.get('/status')
        setStatus(res.data)
      } catch (e) {
        console.error("Backend offline?")
      }
    }

    fetchStatus() // Initial call
    const interval = setInterval(fetchStatus, 1000) // Poll every second
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen pb-24 bg-gray-50 dark:bg-slate-900 text-gray-900 dark:text-gray-100 font-sans transition-colors duration-200">

      {/* Top Navigation Bar */}
      <nav className="fixed top-0 w-full z-50 glass border-b border-white/20 dark:border-slate-700/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/30">
                <Music size={18} fill="currentColor" />
              </div>
              <span className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400">
                DCM Player
              </span>
            </div>

            <div className="flex items-center gap-4">
              <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors">
                <Settings size={20} className="text-gray-500 dark:text-gray-400" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="pt-24 px-4 max-w-5xl mx-auto space-y-8">

        {/* Connection Status Warning */}
        {status === null && (
          <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 flex items-center gap-3 animate-pulse">
            <RefreshCw className="animate-spin" size={20} />
            <span>Connecting to Local Backend... (Run `uvicorn dcm.web_app:app --reload`)</span>
          </div>
        )}

        {/* Hero / Now Playing Section (when not playing, shows welcome) */}
        {!status?.current_song && (
          <div className="glass-panel p-8 rounded-2xl text-center space-y-4 py-16">
            <div className="w-20 h-20 mx-auto rounded-full bg-blue-50 dark:bg-slate-800 flex items-center justify-center text-blue-500">
              <ListMusic size={40} />
            </div>
            <h2 className="text-3xl font-bold">Your Library</h2>
            <p className="text-gray-500 dark:text-gray-400 max-w-md mx-auto">
              Select a song from your file system to start the magic.
              DCM will analyze it and play similar tracks automatically.
            </p>
            <div className="flex justify-center gap-4 pt-4">
              <input
                type="text"
                placeholder="/path/to/song.mp3"
                className="px-4 py-2 rounded-lg border border-gray-300 dark:border-slate-700 bg-white dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 outline-none w-96"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    axios.post(`/play?file_path=${encodeURIComponent(e.target.value)}`)
                  }
                }}
              />
            </div>
          </div>
        )}

        {/* Recommendations Grid */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold flex items-center gap-2">
              <ListMusic className="text-blue-500" />
              Recommended For You
            </h3>
            {status?.current_song && (
              <button
                onClick={() => axios.post(`/recommend?file_path=${encodeURIComponent(status.current_song.file_path)}`)}
                className="text-sm text-blue-500 hover:underline"
              >
                Refresh
              </button>
            )}
          </div>

          <RecommendationList
            currentFile={status?.current_song?.file_path}
            backendUrl={axios.defaults.baseURL}
          />
        </div>

      </main>

      {/* Floating Player Bar */}
      <Player status={status} />

    </div>
  )
}

export default App
