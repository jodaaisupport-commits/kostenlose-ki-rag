package de.kostenlose.kirag.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Verschlüsselte Ablage für API-Keys (Groq/Google) mittels
 * EncryptedSharedPreferences (AES256-GCM, Schlüssel im Android Keystore).
 * Deutlich sicherer als Klartext-SharedPreferences, da die Keys den
 * Nutzer-eigenen API-Zugang zu kostenpflichtigen (wenn auch Free-Tier-)
 * Diensten darstellen.
 */
class SecurePrefs(context: Context) {

    private val prefs: SharedPreferences by lazy {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()

        EncryptedSharedPreferences.create(
            context,
            "secure_prefs",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    var groqApiKey: String
        get() = prefs.getString(KEY_GROQ, "") ?: ""
        set(value) = prefs.edit().putString(KEY_GROQ, value).apply()

    var googleApiKey: String
        get() = prefs.getString(KEY_GOOGLE, "") ?: ""
        set(value) = prefs.edit().putString(KEY_GOOGLE, value).apply()

    companion object {
        private const val KEY_GROQ = "groq_api_key"
        private const val KEY_GOOGLE = "google_api_key"
    }
}
