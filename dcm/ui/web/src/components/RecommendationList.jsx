import { useState, useEffect } from 'react'
import axios from 'axios'
import { PlayCircle, Clock } from 'lucide-react'

const RecommendationList = ({ currentFile, backendUrl }) => {
    const [songs, setSongs] = useState([])
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        if (!currentFile) return

        const fetchRecs = async () => {
            setLoading(true)
            try {
                const res = await axios.post(`/recommend?file_path=${encodeURIComponent(currentFile)}`)
                setSongs(res.data)
            } catch (e) {
                console.error("Failed to get recs", e)
            } finally {
                setLoading(false)
            }
        }

        fetchRecs()
    }, [currentFile])


    if (!currentFile) return (
        <div className="text-center py-10 text-gray-400">
            Play a song to see recommendations
        </div>
    )

    if (loading) return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5].map(i => (
                <div key={i} className="h-20 bg-gray-200 dark:bg-slate-800 rounded-xl animate-pulse" />
            ))}
        </div>
    )

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {songs.map((song, idx) => (
                <div
                    key={idx}
                    className="group glass-panel p-3 rounded-xl flex items-center gap-3 hover:bg-white dark:hover:bg-slate-800 transition-colors cursor-pointer border-transparent hover:border-blue-500/30"
                    onClick={() => axios.post(`/play?file_path=${encodeURIComponent(song.file_path)}`)}
                >
                    {/* Cover Placeholder */}
                    <div className="relative w-12 h-12 rounded-lg bg-gray-100 dark:bg-slate-900 overflow-hidden flex-shrink-0">
                        <div className="w-full h-full bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                            <span className="text-xs font-bold text-blue-500">
                                {song.title.substring(0, 2).toUpperCase()}
                            </span>
                        </div>
                        <div className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity">
                            <PlayCircle className="text-white drop-shadow-lg" size={20} />
                        </div>
                    </div>

                    <div className="min-w-0 flex-1">
                        <h5 className="font-semibold text-sm truncate group-hover:text-blue-500 transition-colors">
                            {song.title}
                        </h5>
                        <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                            {song.artist}
                        </p>
                    </div>

                    {/* Similarity Score Badge */}
                    <div className="text-xs font-mono text-blue-500 bg-blue-50 dark:bg-blue-900/30 px-2 py-1 rounded-md">
                        {Math.round(song.similarity * 100)}%
                    </div>

                </div>
            ))}
        </div>
    )
}

export default RecommendationList
