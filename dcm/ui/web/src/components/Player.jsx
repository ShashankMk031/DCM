import { Play, Pause, SkipBack, SkipForward, Volume2, Sparkles } from 'lucide-react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'

const Player = ({ status }) => {
    if (!status?.current_song) return null

    const { is_playing, current_song, duration, position } = status
    const progress = duration > 0 ? (position / duration) * 100 : 0

    const handleControl = (action) => {
        axios.post(`/control/${action}`)
    }

    const toggleAutoQueue = () => {
        axios.post(`/toggle_auto_queue?enabled=${!status.auto_queue}`)
    }

    return (
        <AnimatePresence>
            <motion.div
                initial={{ y: 100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 100, opacity: 0 }}
                className="fixed bottom-0 w-full z-40 px-4 pb-4"
            >
                <div className="max-w-5xl mx-auto glass p-4 rounded-2xl flex items-center gap-6 shadow-2xl shadow-blue-900/10">

                    {/* Album Art */}
                    <div className="relative group">
                        <div className="w-16 h-16 rounded-xl overflow-hidden bg-gray-200 dark:bg-slate-700 shadow-md">
                            <img
                                src={`http://127.0.0.1:8000/album_art?file_path=${encodeURIComponent(current_song.file_path)}`.replace('/album_art', '/album_art_proxy_todo')}
                                // TODO: Use actual album art endpoint properly
                                onError={(e) => {
                                    e.target.style.display = 'none';
                                    e.target.parentNode.classList.add('flex', 'items-center', 'justify-center', 'bg-gray-800');
                                    e.target.parentNode.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" class="text-gray-400" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>'
                                }}
                                alt={current_song.title}
                                className="w-full h-full object-cover"
                            />
                            <div className="absolute inset-0 bg-black/10 group-hover:bg-black/20 transition-colors" />
                        </div>
                    </div>

                    {/* Song Info */}
                    <div className="flex-1 min-w-0">
                        <h4 className="font-bold text-lg truncate">{current_song.title}</h4>
                        <p className="text-sm text-gray-500 dark:text-gray-400 truncate">
                            {current_song.artist} • {current_song.album}
                        </p>

                        {/* Progress Bar (Visual Only for now) */}
                        <div className="mt-2 h-1 w-full bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 rounded-full transition-all duration-1000 ease-linear"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    </div>

                    {/* Controls */}
                    <div className="flex items-center gap-4">
                        <button
                            onClick={toggleAutoQueue}
                            className={`p-2 rounded-full transition-colors ${status.auto_queue ? 'text-blue-500 bg-blue-100 dark:bg-blue-900/30' : 'text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-700'}`}
                            title="Smart Auto-Queue (Play Similar Songs)"
                        >
                            <Sparkles size={20} />
                        </button>

                        <button
                            onClick={() => handleControl('prev')}
                            className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-full text-gray-600 dark:text-gray-300 transition-colors"
                        >
                            <SkipBack size={24} />
                        </button>

                        <button
                            onClick={() => handleControl(is_playing ? 'pause' : 'resume')}
                            className="w-12 h-12 bg-blue-600 hover:bg-blue-700 text-white rounded-full flex items-center justify-center shadow-lg shadow-blue-500/30 transition-transform active:scale-95"
                        >
                            {is_playing ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" className="ml-1" />}
                        </button>

                        <button
                            onClick={() => handleControl('next')}
                            className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-full text-gray-600 dark:text-gray-300 transition-colors"
                        >
                            <SkipForward size={24} />
                        </button>
                    </div>

                    {/* Volume (Static for UI demo) */}
                    <div className="hidden sm:flex items-center gap-2 text-gray-400">
                        <Volume2 size={20} />
                        <div className="w-20 h-1 bg-gray-200 dark:bg-slate-700 rounded-full">
                            <div className="w-[80%] h-full bg-gray-400 rounded-full" />
                        </div>
                    </div>

                </div>
            </motion.div>
        </AnimatePresence>
    )
}

export default Player
