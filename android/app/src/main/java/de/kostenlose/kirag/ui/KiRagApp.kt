package de.kostenlose.kirag.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import de.kostenlose.kirag.ChatViewModel

private enum class AppTab(val label: String) {
    CHAT("Chat"),
    DOCUMENTS("Dokumente"),
    SETTINGS("Einstellungen"),
}

@Composable
fun KiRagApp(viewModel: ChatViewModel) {
    var selectedTab by rememberSaveable { mutableIntStateOf(0) }
    val uiState by viewModel.uiState.collectAsState()

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    icon = { Icon(Icons.Filled.Chat, contentDescription = null) },
                    label = { Text(AppTab.CHAT.label) },
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    icon = { Icon(Icons.Filled.Description, contentDescription = null) },
                    label = { Text(AppTab.DOCUMENTS.label) },
                )
                NavigationBarItem(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    icon = { Icon(Icons.Filled.Settings, contentDescription = null) },
                    label = { Text(AppTab.SETTINGS.label) },
                )
            }
        },
    ) { innerPadding ->
        Column(modifier = Modifier.padding(innerPadding)) {
            when (selectedTab) {
                0 -> ChatScreen(viewModel = viewModel, uiState = uiState)
                1 -> DocumentsScreen(viewModel = viewModel, uiState = uiState)
                2 -> SettingsScreen(viewModel = viewModel, uiState = uiState)
            }
        }
    }
}
