# PDFBox-Android benötigt diese Regeln, um Reflection-basierte Klassen zu erhalten.
-keep class com.tom_roush.pdfbox.** { *; }
-keep class com.tom_roush.harmony.** { *; }
-dontwarn com.tom_roush.**

# kotlinx.serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}
-keep,includedescriptorclasses class de.kostenlose.kirag.**$$serializer { *; }
-keepclassmembers class de.kostenlose.kirag.** {
    *** Companion;
}
-keepclasseswithmembers class de.kostenlose.kirag.** {
    kotlinx.serialization.KSerializer serializer(...);
}
