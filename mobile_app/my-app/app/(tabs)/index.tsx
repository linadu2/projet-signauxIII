import React, { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  SafeAreaView,
  ScrollView,
  StatusBar,
  Alert,
  Platform,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
// On utilise les icônes incluses par défaut dans Expo
import { Ionicons, MaterialCommunityIcons, Feather } from "@expo/vector-icons";

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const API_URL = "http://192.168.0.30:5000";

  // --- 1. Gestion de la Caméra et Galerie ---

  const pickImage = async () => {
    // Demander la permission d'accéder à la galerie
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3], // Format un peu rectangulaire
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    // Demander la permission d'utiliser la caméra
    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (permission.granted === false) {
      Alert.alert(
          "Permission refusée",
          "Vous devez autoriser la caméra pour prendre une photo."
      );
      return;
    }

    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const resetApp = () => {
    setSelectedImage(null);
    setResult(null);
  };

      // --- 2. Appel API Python (Corrected) ---
  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setResult(null);

    try {
      // A. PREPARE IMAGE
      const localUri = selectedImage;
      const filename = localUri.split('/').pop() || "upload.jpg";
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : `image/jpeg`;

      const formData = new FormData();
      // @ts-ignore
      formData.append('photo', { uri: localUri, name: filename, type });

      // B. START JOB
      const uploadResponse = await fetch(`${API_URL}/analyze-async`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) throw new Error(`Upload failed: ${uploadResponse.status}`);

      const uploadData = await uploadResponse.json();
      const jobId = uploadData.job_id;

      if (!jobId) throw new Error("No job_id returned");

      // C. POLL STATUS (Recursive)
      const pollStatus = async () => {
        try {
          const statusResponse = await fetch(`${API_URL}/status/${jobId}`);
          const job = await statusResponse.json();

          if (job.state === 'done') {
            // SUCCESS: The actual data is inside 'job.result'
            setResult(job.result);
            setIsAnalyzing(false);
          }
          else if (job.state === 'failed') {
            Alert.alert("Erreur", job.error || "Le traitement a échoué");
            setIsAnalyzing(false);
          }
          else {
            // If 'processing' or 'pending', wait 1s and check again
            setTimeout(pollStatus, 1000);
          }
        } catch (err) {
          console.error(err);
          setIsAnalyzing(false); // Stop the spinner on network error
        }
      };

      pollStatus();

    } catch (error) {
      console.error(error);
      Alert.alert("Erreur", "Impossible d'envoyer l'image");
      setIsAnalyzing(false);
    }
  };



  // --- 3. Rendu Visuel ---

  return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#4F46E5" />

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerContent}>
            <MaterialCommunityIcons name="resistor" size={28} color="#FDE047" />
            <Text style={styles.headerTitle}>ResistorScan</Text>
          </View>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>v1.0</Text>
          </View>
        </View>

        <ScrollView contentContainerStyle={styles.scrollContent}>
          {/* Carte Principale */}
          <View style={styles.card}>
            {!selectedImage ? (
                // État initial : Pas d'image
                <View style={styles.placeholderContainer}>
                  <View style={styles.iconCircle}>
                    <Ionicons name="camera" size={48} color="#6366F1" />
                  </View>
                  <Text style={styles.placeholderTitle}>
                    Scanner une résistance
                  </Text>
                  <Text style={styles.placeholderText}>
                    Prenez une photo ou importez une image pour analyser les bandes
                    de couleur.
                  </Text>

                  <View style={styles.buttonRow}>
                    <TouchableOpacity
                        style={styles.actionButtonPrimary}
                        onPress={takePhoto}
                    >
                      <Ionicons name="camera-outline" size={24} color="white" />
                      <Text style={styles.actionButtonTextPrimary}>Caméra</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={styles.actionButtonSecondary}
                        onPress={pickImage}
                    >
                      <Ionicons name="images-outline" size={24} color="#4F46E5" />
                      <Text style={styles.actionButtonTextSecondary}>Galerie</Text>
                    </TouchableOpacity>
                  </View>
                </View>
            ) : (
                // État Image sélectionnée
                <View style={styles.previewContainer}>
                  <View style={styles.imageWrapper}>
                    <Image
                        source={{ uri: selectedImage }}
                        style={styles.previewImage}
                    />
                    <TouchableOpacity style={styles.resetButton} onPress={resetApp}>
                      <Feather name="refresh-cw" size={20} color="white" />
                    </TouchableOpacity>
                  </View>

                  {!result && (
                      <TouchableOpacity
                          style={[
                            styles.analyzeButton,
                            isAnalyzing && styles.analyzeButtonDisabled,
                          ]}
                          onPress={analyzeImage}
                          disabled={isAnalyzing}
                      >
                        {isAnalyzing ? (
                            <ActivityIndicator color="white" />
                        ) : (
                            <>
                              <MaterialCommunityIcons
                                  name="flash"
                                  size={24}
                                  color="white"
                              />
                              <Text style={styles.analyzeButtonText}>
                                Calculer la valeur
                              </Text>
                            </>
                        )}
                      </TouchableOpacity>
                  )}
                </View>
            )}
          </View>

          {/* Carte Résultat */}
          {result && (
              <View style={styles.resultCard}>
                <View style={styles.resultHeader}>
                  <View>
                    <Text style={styles.resultLabel}>VALEUR DÉTECTÉE</Text>
                    <View style={styles.resultValueContainer}>
                      <Text style={styles.resultValue}>{result.resistance}</Text>
                      <Text style={styles.resultUnit}>{result.unit}</Text>
                    </View>
                  </View>
                  <View style={styles.confidenceBadge}>
                    <Feather name="check" size={14} color="#15803D" />
                    <Text style={styles.confidenceText}>Confiance</Text>
                  </View>
                </View>

                <View style={styles.divider} />

                <View style={styles.detailsRow}>
                  <Text style={styles.detailLabel}>Tolérance</Text>
                  <Text style={styles.detailValue}>{result.tolerance}</Text>
                </View>

                <View style={styles.colorsContainer}>
                  <Text style={styles.detailLabel}>Bandes identifiées</Text>
                  <View style={styles.colorBandsRow}>
                    {result.colors.map((color, index) => {
                      // Petit helper pour les couleurs (très basique)
                      const getColorHex = (c) => {
                        const map = {
                          Rouge: "#EF4444",
                          Or: "#EAB308",
                          Marron: "#78350F",
                          Noir: "#000",
                          Vert: "#22C55E",
                          Bleu: "#3B82F6",
                        };
                        return map[c] || "#9CA3AF";
                      };
                      return (
                          <View key={index} style={styles.colorBandWrapper}>
                            <View
                                style={[
                                  styles.colorDot,
                                  { backgroundColor: getColorHex(color) },
                                ]}
                            />
                            <Text style={styles.colorName}>{color}</Text>
                          </View>
                      );
                    })}
                  </View>
                </View>

                <TouchableOpacity style={styles.newScanButton} onPress={resetApp}>
                  <Text style={styles.newScanText}>Scanner une autre</Text>
                </TouchableOpacity>
              </View>
          )}
        </ScrollView>
      </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F3F4F6", // gray-50 equivalent
    paddingTop: Platform.OS === "android" ? StatusBar.currentHeight : 0,
  },
  header: {
    backgroundColor: "#4F46E5", // indigo-600
    padding: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    elevation: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  headerContent: {
    flexDirection: "row",
    alignItems: "center",
  },
  headerTitle: {
    color: "white",
    fontSize: 20,
    fontWeight: "bold",
    marginLeft: 8,
  },
  badge: {
    backgroundColor: "rgba(255,255,255,0.2)",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
  },
  badgeText: {
    color: "white",
    fontSize: 12,
    fontWeight: "600",
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  card: {
    backgroundColor: "white",
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  placeholderContainer: {
    alignItems: "center",
    paddingVertical: 24,
  },
  iconCircle: {
    backgroundColor: "#EEF2FF", // indigo-50
    padding: 20,
    borderRadius: 50,
    marginBottom: 16,
  },
  placeholderTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#1F2937",
    marginBottom: 8,
  },
  placeholderText: {
    textAlign: "center",
    color: "#6B7280",
    marginBottom: 24,
    lineHeight: 20,
  },
  buttonRow: {
    flexDirection: "row",
    gap: 12,
    width: "100%",
    justifyContent: "center",
  },
  actionButtonPrimary: {
    backgroundColor: "#4F46E5",
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
    elevation: 3,
  },
  actionButtonTextPrimary: {
    color: "white",
    fontWeight: "600",
    marginLeft: 8,
  },
  actionButtonSecondary: {
    backgroundColor: "white",
    borderWidth: 1,
    borderColor: "#E0E7FF",
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
  },
  actionButtonTextSecondary: {
    color: "#4F46E5",
    fontWeight: "600",
    marginLeft: 8,
  },
  previewContainer: {
    width: "100%",
  },
  imageWrapper: {
    position: "relative",
    width: "100%",
    height: 220,
    backgroundColor: "#111827",
    borderRadius: 12,
    overflow: "hidden",
    marginBottom: 16,
  },
  previewImage: {
    width: "100%",
    height: "100%",
    resizeMode: "contain",
  },
  resetButton: {
    position: "absolute",
    top: 8,
    right: 8,
    backgroundColor: "rgba(0,0,0,0.6)",
    padding: 8,
    borderRadius: 20,
  },
  analyzeButton: {
    backgroundColor: "#4F46E5",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
    borderRadius: 12,
    elevation: 4,
  },
  analyzeButtonDisabled: {
    backgroundColor: "#9CA3AF",
  },
  analyzeButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
    marginLeft: 8,
  },
  resultCard: {
    backgroundColor: "white",
    borderRadius: 16,
    padding: 20,
    borderLeftWidth: 5,
    borderLeftColor: "#22C55E", // green-500
    elevation: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  resultHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 12,
  },
  resultLabel: {
    color: "#6B7280",
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  resultValueContainer: {
    flexDirection: "row",
    alignItems: "baseline",
    marginTop: 4,
  },
  resultValue: {
    fontSize: 32,
    fontWeight: "800",
    color: "#111827",
  },
  resultUnit: {
    fontSize: 20,
    color: "#4B5563",
    marginLeft: 4,
    fontWeight: "600",
  },
  confidenceBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#DCFCE7", // green-100
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  confidenceText: {
    color: "#15803D", // green-700
    fontSize: 10,
    fontWeight: "bold",
    marginLeft: 4,
  },
  divider: {
    height: 1,
    backgroundColor: "#F3F4F6",
    marginVertical: 12,
  },
  detailsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  detailLabel: {
    color: "#6B7280",
    fontSize: 14,
  },
  detailValue: {
    color: "#1F2937",
    fontWeight: "600",
    fontSize: 14,
  },
  colorsContainer: {
    marginTop: 4,
  },
  colorBandsRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 8,
  },
  colorBandWrapper: {
    alignItems: "center",
  },
  colorDot: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    marginBottom: 4,
    elevation: 2,
  },
  colorName: {
    fontSize: 10,
    color: "#6B7280",
  },
  newScanButton: {
    marginTop: 20,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 12,
    alignItems: "center",
  },
  newScanText: {
    color: "#4B5563",
    fontWeight: "600",
  },
});
