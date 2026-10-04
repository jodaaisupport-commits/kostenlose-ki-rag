package de.kostenlose.kirag

import android.app.Application
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader

/** Application-Klasse: initialisiert PDFBox-Android einmalig beim App-Start. */
class KiRagApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        PDFBoxResourceLoader.init(applicationContext)
    }
}
