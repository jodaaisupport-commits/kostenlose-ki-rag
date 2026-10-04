package de.kostenlose.kirag.network

/** Entspricht llm/providers.py::ProviderError – allgemeiner, dem Nutzer
 * anzuzeigender Fehler eines LLM-Providers. */
open class ProviderError(message: String) : Exception(message)

/** Entspricht llm/providers.py::RateLimitError – spezifischer Free-Tier-Rate-Limit-Fehler. */
class RateLimitError(message: String) : ProviderError(message)
