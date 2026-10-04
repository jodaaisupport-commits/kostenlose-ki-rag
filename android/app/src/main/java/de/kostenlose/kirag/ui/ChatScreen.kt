package de.kostenlose.kirag.ui

import android.content.Intent
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import de.kostenlose.kirag.ChatUiState
import de.kostenlose.kirag.ChatViewModel
import de.kostenlose.kirag.network.ChatMessage
import de.kostenlose.kirag.network.ChatRole
import de.kostenlose.kirag.network.GEMINI_MODELS
import de.kostenlose.kirag.network.GROQ_MODELS
import de.kostenlose.kirag.network.LlmProvider
import de.kostenlose.kirag.network.SYSTEM_PROMPT_PRESETS
import de.kostenlose.kirag.rag.SearchResult

@Composable
fun ChatScreen(viewModel: ChatViewModel, uiState: ChatUiState) {
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val context = LocalContext.current

    LaunchedEffect(uiState.messages.size) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Kompakte Modell-/RAG-Statuszeile oberhalb des Chats
        ChatHeaderBar(viewModel = viewModel, uiState = uiState)

        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (uiState.messages.isEmpty()) {
                item { EmptyChatHint() }
            }
            itemsIndexed(uiState.messages) { _, msg ->
                ChatBubble(message = msg)
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Stelle eine Frage ...") },
                maxLines = 4,
            )
            Spacer(modifier = Modifier.width(8.dp))
            if (uiState.isGenerating) {
                IconButton(onClick = { viewModel.stopGeneration() }) {
                    Icon(Icons.Filled.Stop, contentDescription = "Stopp", tint = MaterialTheme.colorScheme.error)
                }
            } else {
                IconButton(
                    onClick = {
                        if (input.isNotBlank()) {
                            viewModel.sendMessage(input)
                            input = ""
                        }
                    },
                ) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Senden")
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            TextButton(onClick = { viewModel.clearChat() }) {
                Icon(Icons.Filled.Clear, contentDescription = null, modifier = Modifier.height(18.dp))
                Spacer(Modifier.width(4.dp))
                Text("Chat leeren")
            }
            TextButton(onClick = {
                val file = viewModel.exportChatToFile() ?: return@TextButton
                val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
                val intent = Intent(Intent.ACTION_SEND).apply {
                    type = "text/markdown"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                context.startActivity(Intent.createChooser(intent, "Chat exportieren"))
            }) {
                Icon(Icons.Filled.Share, contentDescription = null, modifier = Modifier.height(18.dp))
                Spacer(Modifier.width(4.dp))
                Text("Exportieren")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatHeaderBar(viewModel: ChatViewModel, uiState: ChatUiState) {
    var providerExpanded by remember { mutableStateOf(false) }
    var modelExpanded by remember { mutableStateOf(false) }

    Surface(tonalElevation = 2.dp) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            ExposedDropdownMenuBox(
                expanded = providerExpanded,
                onExpandedChange = { providerExpanded = it },
                modifier = Modifier.weight(1f),
            ) {
                OutlinedTextField(
                    value = uiState.provider.displayName,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Anbieter") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = providerExpanded) },
                    modifier = Modifier.menuAnchor().weight(1f),
                )
                ExposedDropdownMenu(
                    expanded = providerExpanded,
                    onDismissRequest = { providerExpanded = false },
                ) {
                    LlmProvider.entries.forEach { p ->
                        androidx.compose.material3.DropdownMenuItem(
                            text = { Text(p.displayName) },
                            onClick = { viewModel.updateProvider(p); providerExpanded = false },
                        )
                    }
                }
            }

            val currentModel = if (uiState.provider == LlmProvider.GROQ) uiState.groqModel else uiState.geminiModel
            val models = if (uiState.provider == LlmProvider.GROQ) GROQ_MODELS else GEMINI_MODELS

            ExposedDropdownMenuBox(
                expanded = modelExpanded,
                onExpandedChange = { modelExpanded = it },
                modifier = Modifier.weight(1f),
            ) {
                OutlinedTextField(
                    value = currentModel,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text("Modell") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = modelExpanded) },
                    modifier = Modifier.menuAnchor().weight(1f),
                )
                ExposedDropdownMenu(
                    expanded = modelExpanded,
                    onDismissRequest = { modelExpanded = false },
                ) {
                    models.forEach { m ->
                        androidx.compose.material3.DropdownMenuItem(
                            text = { Text(m) },
                            onClick = {
                                if (uiState.provider == LlmProvider.GROQ) viewModel.updateGroqModel(m)
                                else viewModel.updateGeminiModel(m)
                                modelExpanded = false
                            },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun EmptyChatHint() {
    Column(
        modifier = Modifier.fillMaxWidth().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "🆓 Kostenlose KI-RAG-Chat-App",
            style = MaterialTheme.typography.titleLarge,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Stelle eine Frage oder lade Dokumente im Tab \"Dokumente\" hoch, " +
                "um mit RAG-Kontext zu chatten.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ChatBubble(message: ChatMessage) {
    val isUser = message.role == ChatRole.USER
    val clipboard = LocalClipboardManager.current
    var showCitations by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Column(
            modifier = Modifier.widthIn(max = 320.dp),
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start,
        ) {
            Card(
                modifier = Modifier.combinedClickable(
                    onClick = {},
                    onLongClick = { clipboard.setText(AnnotatedString(message.content)) },
                ),
                colors = CardDefaults.cardColors(
                    containerColor = if (isUser) MaterialTheme.colorScheme.primaryContainer
                    else MaterialTheme.colorScheme.surfaceVariant,
                ),
                shape = RoundedCornerShape(16.dp),
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    if (message.content.isEmpty()) {
                        CircularProgressIndicator(modifier = Modifier.height(16.dp).width(16.dp), strokeWidth = 2.dp)
                    } else {
                        Text(message.content)
                    }
                }
            }

            if (message.citations.isNotEmpty()) {
                TextButton(onClick = { showCitations = !showCitations }) {
                    Text(
                        if (showCitations) "▲ Quellen verbergen"
                        else "📚 ${message.citations.size} Quelle(n) anzeigen",
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
                if (showCitations) {
                    CitationsBlock(message.citations)
                }
            }
        }
    }
}

@Composable
private fun CitationsBlock(citations: List<SearchResult>) {
    Column(
        modifier = Modifier
            .widthIn(max = 320.dp)
            .padding(start = 8.dp, end = 8.dp, bottom = 8.dp),
    ) {
        citations.forEach { (chunk, score) ->
            Card(
                modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            ) {
                Column(modifier = Modifier.padding(8.dp)) {
                    val pageInfo = chunk.page?.let { ", Seite $it" } ?: ""
                    Text(
                        "${chunk.source}$pageInfo · Relevanz ${"%.2f".format(score)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        chunk.text.take(180) + if (chunk.text.length > 180) " …" else "",
                        style = MaterialTheme.typography.bodyLarge.copy(fontSize = 13.sp),
                    )
                }
            }
        }
    }
}
