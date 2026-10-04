package de.kostenlose.kirag.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import de.kostenlose.kirag.ChatUiState
import de.kostenlose.kirag.ChatViewModel
import de.kostenlose.kirag.network.SYSTEM_PROMPT_PRESETS

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(viewModel: ChatViewModel, uiState: ChatUiState) {
    var presetExpanded by remember { mutableStateOf(false) }
    var selectedPreset by remember { mutableStateOf(SYSTEM_PROMPT_PRESETS.first().label) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("🔑 API-Schlüssel (kostenlos erhältlich)", style = MaterialTheme.typography.titleLarge)
        Text(
            "• Groq: console.groq.com/keys\n• Google Gemini: aistudio.google.com/apikey",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        OutlinedTextField(
            value = uiState.groqApiKey,
            onValueChange = { viewModel.updateGroqApiKey(it) },
            label = { Text("Groq API-Key") },
            visualTransformation = PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = uiState.googleApiKey,
            onValueChange = { viewModel.updateGoogleApiKey(it) },
            label = { Text("Google (Gemini) API-Key") },
            visualTransformation = PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
            modifier = Modifier.fillMaxWidth(),
        )
        Text(
            "🔒 Die Schlüssel werden verschlüsselt (AES-256) ausschließlich lokal " +
                "auf deinem Gerät gespeichert und nie an Dritte außer den jeweiligen " +
                "API-Anbieter übertragen.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Divider()

        Text("🤖 Modell-Einstellungen", style = MaterialTheme.typography.titleLarge)

        Text("Temperature: ${"%.2f".format(uiState.temperature)}")
        Slider(
            value = uiState.temperature.toFloat(),
            onValueChange = { viewModel.updateTemperature(it.toDouble()) },
            valueRange = 0f..1.5f,
        )

        ExposedDropdownMenuBox(
            expanded = presetExpanded,
            onExpandedChange = { presetExpanded = it },
        ) {
            OutlinedTextField(
                value = selectedPreset,
                onValueChange = {},
                readOnly = true,
                label = { Text("System-Prompt-Vorlage") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = presetExpanded) },
                modifier = Modifier.menuAnchor().fillMaxWidth(),
            )
            ExposedDropdownMenu(
                expanded = presetExpanded,
                onDismissRequest = { presetExpanded = false },
            ) {
                SYSTEM_PROMPT_PRESETS.forEach { preset ->
                    androidx.compose.material3.DropdownMenuItem(
                        text = { Text(preset.label) },
                        onClick = {
                            selectedPreset = preset.label
                            viewModel.updateSystemPrompt(preset.prompt)
                            presetExpanded = false
                        },
                    )
                }
            }
        }

        OutlinedTextField(
            value = uiState.systemPrompt,
            onValueChange = { viewModel.updateSystemPrompt(it) },
            label = { Text("System-Prompt") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 4,
            maxLines = 8,
        )

        Card {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("ℹ️ Über diese App", style = MaterialTheme.typography.titleMedium)
                Text(
                    "Kostenlose KI-RAG-Chat-App – native Android-Portierung der " +
                        "Gradio-Webanwendung. Chat läuft über die Free-Tier-APIs von " +
                        "Groq/Google Gemini, die Dokumentensuche (TF-IDF) läuft " +
                        "komplett offline auf diesem Gerät.",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    }
}
