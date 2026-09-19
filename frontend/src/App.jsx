import { useEffect, useMemo, useState } from "react"
import { Search, User, Star, Sparkles, LoaderCircle } from "lucide-react"
import {
  createNewProfile,
  getExistingHistory,
  getExistingRecommendations,
  getExistingUser,
  getGenres,
  getMovies,
  getNewHistory,
  getNewRecommendations,
  rateExistingUser,
  rateNewProfile,
} from "./api/client"

function Stars({ value = 0, onChange, readonly = false }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          type="button"
          disabled={readonly}
          onClick={() => onChange?.(star)}
          className="cursor-pointer disabled:cursor-default"
          aria-label={`${star} stars`}
        >
          <Star
            size={20}
            className={star <= value ? "fill-current" : ""}
          />
        </button>
      ))}
    </div>
  )
}

function MovieCard({ movie, ratedValue, onOpen, showScore = false }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(movie)}
      className="cursor-pointer text-left rounded-2xl bg-white border border-gray-200 overflow-hidden hover:-translate-y-0.5 transition"
    >
      <div className="relative aspect-[2/3] bg-gray-100">
        {movie.poster_url ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full grid place-items-center text-sm text-gray-400">
            No poster
          </div>
        )}
        {showScore && Number.isFinite(movie.score) ? (
          <div className="absolute top-2 right-2 rounded-lg bg-black/80 px-2 py-1 text-xs font-semibold text-white shadow">
            Score {movie.score.toFixed(3)}
          </div>
        ) : null}
      </div>

      <div className="p-3 space-y-2">
        <div className="font-semibold line-clamp-2">{movie.title}</div>
        <div className="text-xs text-gray-500 line-clamp-1">
          {movie.genres?.join(" • ")}
        </div>

        {ratedValue ? (
          <div className="text-sm">Your rating: {ratedValue}/5</div>
        ) : null}

        {movie.rank && !showScore ? (
          <div className="text-xs text-gray-500">
            #{movie.rank} · score {movie.score?.toFixed?.(3)}
          </div>
        ) : null}
      </div>
    </button>
  )
}

function MovieGridLoading({ count = 6 }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-4" aria-label="Loading movies">
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="overflow-hidden rounded-2xl border border-gray-200 bg-white">
          <div className="aspect-[2/3] animate-pulse bg-gray-200" />
          <div className="space-y-2 p-3">
            <div className="h-4 animate-pulse rounded bg-gray-200" />
            <div className="h-3 w-2/3 animate-pulse rounded bg-gray-100" />
          </div>
        </div>
      ))}
    </div>
  )
}

function App() {
  const [mode, setMode] = useState(null)
  const [existingId, setExistingId] = useState("")
  const [subject, setSubject] = useState(null)

  const [tab, setTab] = useState("movies")
  const [movies, setMovies] = useState([])
  const [genres, setGenres] = useState([])
  const [selectedGenres, setSelectedGenres] = useState([])
  const [query, setQuery] = useState("")
  const [year, setYear] = useState("")
  const [history, setHistory] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [selectedMovie, setSelectedMovie] = useState(null)
  const [profileOpen, setProfileOpen] = useState(false)
  const [moviesLoading, setMoviesLoading] = useState(false)
  const [recommendationsLoading, setRecommendationsLoading] = useState(false)
  const [onboardingLoading, setOnboardingLoading] = useState(false)

  const [onboardingRatings, setOnboardingRatings] = useState({})
  const [onboardingQuery, setOnboardingQuery] = useState("")
  const [onboardingMovies, setOnboardingMovies] = useState([])

  const historyMap = useMemo(
    () => Object.fromEntries(history.map((x) => [x.movie_id, x.rating])),
    [history],
  )

  async function refreshHistory(current = subject) {
    if (!current) return
    const data = current.type === "existing"
      ? await getExistingHistory(current.id)
      : await getNewHistory(current.id)
    setHistory(data.ratings || [])
  }

  async function loadMovies() {
    setMoviesLoading(true)
    try {
      const data = await getMovies({
        query,
        genres: selectedGenres,
        year: year || undefined,
        limit: 24,
      })
      setMovies(data)
    } finally {
      setMoviesLoading(false)
    }
  }

  useEffect(() => {
    getGenres().then(setGenres).catch(console.error)
  }, [])

  async function enterExisting() {
    const id = Number(existingId)
    if (!id) return
    await getExistingUser(id)
    const current = { type: "existing", id }
    setSubject(current)
    setMode(null)
    const data = await getExistingHistory(id)
    setHistory(data.ratings || [])
    await loadMovies()
  }

  async function searchOnboarding() {
    setOnboardingLoading(true)
    try {
      const data = await getMovies({
        query: onboardingQuery,
        limit: 20,
      })
      setOnboardingMovies(data)
    } finally {
      setOnboardingLoading(false)
    }
  }

  async function finishNewUser() {
    const entries = Object.entries(onboardingRatings)
    if (entries.length < 5) return

    const payload = entries.map(([movie_id, rating]) => ({
      movie_id: Number(movie_id),
      rating: Number(rating),
    }))

    const created = await createNewProfile(payload)
    const current = { type: "new", id: created.profile_id }
    setSubject(current)
    setMode(null)
    await refreshHistory(current)
    await loadMovies()
  }

  async function recommend() {
    if (!subject) return
    setRecommendationsLoading(true)
    try {
      const data = subject.type === "existing"
        ? await getExistingRecommendations(subject.id)
        : await getNewRecommendations(subject.id)
      setRecommendations(data.recommendations || [])
    } finally {
      setRecommendationsLoading(false)
    }
  }

  async function rateMovie(movieId, ratingValue) {
    if (!subject) return

    if (subject.type === "existing") {
      await rateExistingUser(subject.id, movieId, ratingValue)
    } else {
      await rateNewProfile(subject.id, movieId, ratingValue)
    }

    await refreshHistory()
    setSelectedMovie(null)
  }

  function toggleGenre(genre) {
    setSelectedGenres((current) =>
      current.includes(genre)
        ? current.filter((g) => g !== genre)
        : [...current, genre]
    )
  }

  if (!subject) {
    return (
      <main className="min-h-screen grid place-items-center p-6">
        <div className="w-full max-w-3xl">
          {!mode ? (
            <div className="grid md:grid-cols-2 gap-4">
              <button
                type="button"
                onClick={() => setMode("existing")}
                className="cursor-pointer rounded-3xl border bg-white p-8 text-left"
              >
                <User className="mb-4" />
                <h2 className="text-2xl font-bold">Existing User</h2>
                <p className="text-gray-500 mt-2">Enter a MovieLens user ID.</p>
              </button>

              <button
                type="button"
                onClick={() => setMode("new")}
                className="cursor-pointer rounded-3xl border bg-white p-8 text-left"
              >
                <Sparkles className="mb-4" />
                <h2 className="text-2xl font-bold">New User</h2>
                <p className="text-gray-500 mt-2">Rate at least 5 movies.</p>
              </button>
            </div>
          ) : null}

          {mode === "existing" ? (
            <div className="rounded-3xl border bg-white p-6 space-y-4">
              <button type="button" onClick={() => setMode(null)} className="cursor-pointer text-sm">
                ← Back
              </button>
              <h2 className="text-2xl font-bold">Existing User</h2>
              <input
                className="w-full rounded-xl border px-4 py-3"
                value={existingId}
                onChange={(e) => setExistingId(e.target.value)}
                placeholder="User ID, e.g. 128"
              />
              <button
                type="button"
                onClick={enterExisting}
                className="cursor-pointer rounded-xl bg-black text-white px-5 py-3"
              >
                Continue
              </button>
            </div>
          ) : null}

          {mode === "new" ? (
            <div className="rounded-3xl border bg-white p-6 space-y-4">
              <div className="flex items-center justify-between">
                <button type="button" onClick={() => setMode(null)} className="cursor-pointer text-sm">
                  ← Back
                </button>
                <span className="text-sm">
                  {Object.keys(onboardingRatings).length}/5 rated
                </span>
              </div>

              <h2 className="text-2xl font-bold">Rate at least 5 movies</h2>

              <div className="flex gap-2">
                <input
                  className="flex-1 rounded-xl border px-4 py-3"
                  value={onboardingQuery}
                  onChange={(e) => setOnboardingQuery(e.target.value)}
                  placeholder="Search movie name"
                />
                <button
                  type="button"
                  onClick={searchOnboarding}
                  className="cursor-pointer rounded-xl border px-4"
                >
                  <Search size={18} />
                </button>
              </div>

              <div className="space-y-2 max-h-[420px] overflow-auto">
                {onboardingLoading ? (
                  <div className="flex justify-center py-10 text-gray-500">
                    <LoaderCircle size={24} className="animate-spin" aria-label="Loading movies" />
                  </div>
                ) : onboardingMovies.map((movie) => (
                  <div
                    key={movie.movie_id}
                    className="flex items-center justify-between gap-4 rounded-xl border p-3"
                  >
                    <div className="min-w-0">
                      <div className="font-medium truncate">{movie.title}</div>
                      <div className="text-xs text-gray-500">
                        {movie.genres?.join(" • ")}
                      </div>
                    </div>
                    <Stars
                      value={onboardingRatings[movie.movie_id] || 0}
                      onChange={(value) =>
                        setOnboardingRatings((current) => ({
                          ...current,
                          [movie.movie_id]: value,
                        }))
                      }
                    />
                  </div>
                ))}
              </div>

              <button
                type="button"
                disabled={Object.keys(onboardingRatings).length < 5}
                onClick={finishNewUser}
                className="cursor-pointer rounded-xl bg-black text-white px-5 py-3 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Continue
              </button>
            </div>
          ) : null}
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gray-50">
      <header className="sticky top-0 z-20 border-b bg-white">
        <div className="max-w-7xl mx-auto px-5 py-4 flex items-center justify-between gap-4">
          <div className="cursor-pointer font-bold text-xl">MovieLens DSS</div>

          <nav className="flex gap-2">
            <button
              type="button"
              onClick={() => setTab("movies")}
              className={`cursor-pointer px-4 py-2 rounded-xl ${tab === "movies" ? "bg-black text-white" : "bg-gray-100"}`}
            >
              Movies
            </button>
            <button
              type="button"
              onClick={() => setTab("recommendations")}
              className={`cursor-pointer px-4 py-2 rounded-xl ${tab === "recommendations" ? "bg-black text-white" : "bg-gray-100"}`}
            >
              Recommendations
            </button>
          </nav>

          <button
            type="button"
            onClick={() => setProfileOpen(true)}
            className="cursor-pointer rounded-full border p-2"
          >
            <User size={20} />
          </button>
        </div>
      </header>

      <section className="max-w-7xl mx-auto px-5 py-6">
        {tab === "movies" ? (
          <div className="space-y-5">
            <div className="grid md:grid-cols-[1fr_auto] gap-3">
              <div className="flex gap-2">
                <input
                  className="flex-1 rounded-xl border px-4 py-3 bg-white"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search movie title"
                />
                <input
                  className="w-36 rounded-xl border px-4 py-3 bg-white"
                  type="number"
                  min="1900"
                  max="2003"
                  value={year}
                  onChange={(e) => setYear(e.target.value)}
                  placeholder="Year"
                />
              </div>

              <button
                type="button"
                onClick={loadMovies}
                className="cursor-pointer rounded-xl bg-black text-white px-5 py-3"
              >
                Search
              </button>
            </div>

            <div className="flex flex-wrap gap-2">
              {genres.map((genre) => (
                <button
                  key={genre}
                  type="button"
                  onClick={() => toggleGenre(genre)}
                  className={`cursor-pointer rounded-full border px-3 py-1.5 text-sm ${
                    selectedGenres.includes(genre)
                      ? "bg-black text-white"
                      : "bg-white"
                  }`}
                >
                  {genre}
                </button>
              ))}
            </div>

            {moviesLoading ? <MovieGridLoading count={12} /> : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-4">
                {movies.map((movie) => (
                  <MovieCard
                    key={movie.movie_id}
                    movie={movie}
                    ratedValue={historyMap[movie.movie_id]}
                    onOpen={setSelectedMovie}
                  />
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Top 10 Recommendations</h2>
                <p className="text-gray-500">NCF personalized ranking</p>
              </div>
              <button
                type="button"
                onClick={recommend}
                className="cursor-pointer rounded-xl bg-black text-white px-5 py-3"
              >
                {recommendationsLoading ? (
                  <span className="flex items-center gap-2">
                    <LoaderCircle size={18} className="animate-spin" aria-hidden="true" />
                    Loading...
                  </span>
                ) : "Recommend"}
              </button>
            </div>

            {recommendationsLoading ? <MovieGridLoading count={10} /> : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                {recommendations.map((movie) => (
                  <MovieCard
                    key={movie.movie_id}
                    movie={movie}
                    ratedValue={historyMap[movie.movie_id]}
                    onOpen={setSelectedMovie}
                    showScore
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </section>

      {selectedMovie ? (
        <div className="fixed inset-0 z-30 bg-black/40 grid place-items-center p-4">
          <div className="w-full max-w-2xl rounded-3xl bg-white overflow-hidden">
            {selectedMovie.backdrop_url ? (
              <img
                src={selectedMovie.backdrop_url}
                alt=""
                className="w-full aspect-[16/6] object-cover"
              />
            ) : null}

            <div className="p-6 space-y-4">
              <div className="flex justify-between gap-4">
                <div>
                  <h3 className="text-2xl font-bold">{selectedMovie.title}</h3>
                  <p className="text-sm text-gray-500">
                    {selectedMovie.genres?.join(" • ")}
                  </p>
                </div>
                <button type="button" className="cursor-pointer" onClick={() => setSelectedMovie(null)}>
                  ✕
                </button>
              </div>

              {selectedMovie.overview ? (
                <p className="text-gray-600">{selectedMovie.overview}</p>
              ) : null}

              {historyMap[selectedMovie.movie_id] ? (
                <div>
                  <div className="text-sm mb-2">Your rating</div>
                  <Stars value={historyMap[selectedMovie.movie_id]} readonly />
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-sm">Rate this film</div>
                  <Stars onChange={(value) => rateMovie(selectedMovie.movie_id, value)} />
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {profileOpen ? (
        <div className="fixed inset-0 z-30 bg-black/40 flex justify-end">
          <aside className="w-full max-w-md h-full bg-white p-6 overflow-auto">
            <div className="flex justify-between mb-6">
              <div>
                <div className="text-sm text-gray-500">User ID</div>
                <div className="font-bold break-all">{subject.id}</div>
              </div>
              <button type="button" className="cursor-pointer" onClick={() => setProfileOpen(false)}>
                ✕
              </button>
            </div>

            <h3 className="font-bold text-lg mb-3">Rated Movies</h3>
            <div className="space-y-3">
              {history.map((movie) => (
                <div key={movie.movie_id} className="border-b pb-3">
                  <div className="font-medium">{movie.title}</div>
                  <div className="text-sm text-gray-500">
                    {movie.rating}/5
                  </div>
                </div>
              ))}
            </div>
          </aside>
        </div>
      ) : null}
    </main>
  )
}

export default App
