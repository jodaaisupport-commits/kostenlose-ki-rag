package de.kostenlose.kirag.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import de.kostenlose.kirag.ChatUiState
import de.kostenlose.kirag.ChatViewModel

@Composable
fun DocumentsScreen(viewModel: ChatViewModel, uiState: ChatUiState) {
    var selectedFileNames by remember { mutableStateOf<List<String>>(emptyList()) }

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments(),
    ) { uris ->
        if (uris.isNotEmpty()) {
            viewModel.indexDocuments(uris)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("📚 Dokumenten-RAG", style = MaterialTheme.typography.titleLarge)
        Text(
            "Lade PDF-, TXT- oder MD-Dateien hoch (max. 20 Dateien, 25 MB/Datei). " +
                "Die Suche läuft komplett lokal auf deinem Gerät (TF-IDF) – " +
                "kostenlos und ohne zusätzlichen API-Key.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Switch(
                checked = uiState.useRag,
                onCheckedChange = { viewModel.updateUseRag(it) },
            )
            Text("RAG-Kontext im Chat verwenden")
        }

        Card {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Chunk-Einstellungen", style = MaterialTheme.typography.titleMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = uiState.chunkSize.toString(),
                        onValueChange = { v -> v.toIntOrNull()?.let { viewModel.updateChunkSize(it) } },
                        label = { Text("Chunk-Größe (Wörter)") },
                        modifier = Modifier.weight(1f),
                    )
                    OutlinedTextField(
                        value = uiState.overlap.toString(),
                        onValueChange = { v -> v.toIntOrNull()?.let { viewModel.updateOverlap(it) } },
                        label = { Text("Überlappung (Wörter)") },
                        modifier = Modifier.weight(1f),
                    )
                }
                Text("Anzahl Kontext-Abschnitte (Top-K): ${uiState.topK}")
                Slider(
                    value = uiState.topK.toFloat(),
                    onValueChange = { viewModel.updateTopK(it.toInt()) },
                    valueRange = 1f..10f,
                    steps = 8,
                )
            }
        }

        Button(
            onClick = { filePicker.launch(arrayOf("application/pdf", "text/plain", "text/markdown")) },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Icon(Icons.Filled.UploadFile, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Dokumente auswählen & indizieren")
        }

        if (uiState.indexInProgress) {
            LinearProgressIndicator(
                progress = { uiState.indexProgress },
                modifier = Modifier.fillMaxWidth(),
            )
        }

        if (uiState.indexStatus.isNotBlank()) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Text(
                    uiState.indexStatus,
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }

        Divider()

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Indizierte Dokumente", style = MaterialTheme.typography.titleLarge)
            OutlinedButton(onClick = { viewModel.clearIndex() }) {
                Icon(Icons.Filled.Delete, contentDescription = null)
                Spacer(Modifier.width(4.dp))
                Text("Index löschen")
            }
        }

        val index = uiState.ragIndex
        if (index == null || !index.isReady) {
            Text(
                "Noch keine Dokumente indiziert.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                index.sources.forEach { src ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Filled.Description,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.height(18.dp),
                        )
                        Spacer(Modifier.width(6.dp))
                        Text("${src.name} (${src.chunkCount} Abschnitte)")
                    }
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    "${index.chunks.size} Text-Abschnitte insgesamt indiziert · Modus: TF-IDF (on-device)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
