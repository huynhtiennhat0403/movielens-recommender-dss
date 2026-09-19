import axios from "axios"

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
  timeout: 15000,
})

export async function getGenres() {
  const { data } = await api.get("/movies/genres")
  return data
}

export async function getMovies(params = {}) {
  const searchParams = new URLSearchParams()

  if (params.query) searchParams.set("query", params.query)
  if (params.year) searchParams.set("year", params.year)
  if (params.limit) searchParams.set("limit", params.limit)
  if (params.offset) searchParams.set("offset", params.offset)

  for (const genre of params.genres || []) {
    searchParams.append("genres", genre)
  }

  const { data } = await api.get(`/movies?${searchParams.toString()}`)
  return data
}

export async function getExistingUser(userId) {
  const { data } = await api.get(`/users/${userId}`)
  return data
}

export async function getExistingHistory(userId) {
  const { data } = await api.get(`/users/${userId}/history`)
  return data
}

export async function getExistingRecommendations(userId) {
  const { data } = await api.get(`/users/${userId}/recommendations?top_k=10`)
  return data
}

export async function rateExistingUser(userId, movieId, rating) {
  const { data } = await api.post(`/users/${userId}/ratings`, {
    movie_id: movieId,
    rating,
  })
  return data
}

export async function createNewProfile(ratings) {
  const { data } = await api.post("/profiles", {
    ratings,
  })
  return data
}

export async function getNewHistory(profileId) {
  const { data } = await api.get(`/profiles/${profileId}/history`)
  return data
}

export async function getNewRecommendations(profileId) {
  const { data } = await api.get(`/profiles/${profileId}/recommendations?top_k=10`)
  return data
}

export async function rateNewProfile(profileId, movieId, rating) {
  const { data } = await api.post(`/profiles/${profileId}/ratings`, {
    movie_id: movieId,
    rating,
  })
  return data
}