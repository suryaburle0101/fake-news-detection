import React, { useState } from "react";
import * as Clipboard from 'expo-clipboard';

import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  ActivityIndicator,
} from "react-native";

export default function HomeScreen() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const checkNews = async () => {
    if (!text) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(
  "https://prosodic-unprodded-jake.ngrok-free.dev/predict",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }
);


      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.log(error);
    }

    setLoading(false);
  };

  const isReal = result?.prediction === "True";
  const handlePaste = async () => {
  const textFromClipboard = await Clipboard.getStringAsync();
  setText(textFromClipboard);
};


  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#f5f6fa" }}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Fake news detector</Text>
        <Text style={styles.subtitle}>
          Designed using Deep Learning Techniques
        </Text>

        {/* Input Card */}
        <View style={styles.card}>
          <TextInput
  placeholder="Enter news text..."
  placeholderTextColor="#000"
  multiline
  style={styles.input}
  value={text}
  onChangeText={setText}
/>

        </View>

        {/* Clear + Paste */}
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={styles.smallButton}
            onPress={() => setText("")}
          >
            <Text style={styles.smallButtonText}>Clear</Text>
          </TouchableOpacity>

          <TouchableOpacity
  style={styles.smallButton}
  onPress={handlePaste}
>

            <Text style={styles.smallButtonText}>Paste</Text>
          </TouchableOpacity>
        </View>

        {/* Analyze Button */}
        <TouchableOpacity style={styles.analyzeButton} onPress={checkNews}>
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.analyzeText}>Analyze</Text>
          )}
        </TouchableOpacity>

        {/* Result */}
        {result && (
          <View
            style={[
              styles.resultCard,
              { borderColor: isReal ? "#2ecc71" : "#e74c3c" },
            ]}
          >
            <Text style={styles.resultTitle}>ANALYSIS RESULT</Text>

            <Text
              style={[
                styles.resultPrediction,
                { color: isReal ? "#2ecc71" : "#e74c3c" },
              ]}
            >
              {isReal ? "Real" : "Fake"}
            </Text>

            <View style={styles.scoreBox}>
              <Text style={styles.scoreText}>
                Credibility Score: {result.confidence}%
              </Text>
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    paddingTop: 30,
  },
  title: {
    fontSize: 26,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 5,
  },
  subtitle: {
    textAlign: "center",
    color: "#777",
    marginBottom: 25,
  },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: 20,
    padding: 20,
    elevation: 3,
  },
  input: {
    minHeight: 120,
    textAlignVertical: "top",
    fontSize: 16,
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 15,
    marginBottom: 20,
  },
  smallButton: {
    backgroundColor: "#eaeaea",
    paddingVertical: 10,
    paddingHorizontal: 25,
    borderRadius: 25,
  },
  smallButtonText: {
    fontWeight: "500",
  },
  analyzeButton: {
    backgroundColor: "#4a90e2",
    padding: 16,
    borderRadius: 15,
    alignItems: "center",
    marginBottom: 25,
  },
  analyzeText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  resultCard: {
    backgroundColor: "#ffffff",
    borderRadius: 20,
    padding: 20,
    borderWidth: 2,
    alignItems: "center",
  },
  resultTitle: {
    color: "#888",
    marginBottom: 10,
  },
  resultPrediction: {
    fontSize: 30,
    fontWeight: "bold",
    marginBottom: 15,
  },
  scoreBox: {
    backgroundColor: "#f1f3f6",
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 15,
  },
  scoreText: {
    fontSize: 14,
  },
});
