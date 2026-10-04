package de.kostenlose.kirag

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import de.kostenlose.kirag.ui.KiRagApp
import de.kostenlose.kirag.ui.theme.KiRagTheme

class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            KiRagTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    KiRagApp(viewModel = viewModel)
                }
            }
        }
    }
}
