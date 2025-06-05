import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [tahmin, setTahmin] = useState("");
  const [ilacBilgisi, setIlacBilgisi] = useState(null);
  const [uyari, setUyari] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setTahmin("");
    setIlacBilgisi(null);
    setUyari("");
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      alert("Lütfen bir görsel seçin.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:8000/kutu-tahmin", formData);
      setTahmin(response.data.tahmin || "Tahmin yapılamadı.");
      setIlacBilgisi(response.data.ilac_bilgisi || null);
      setUyari(response.data.uyari || "");  // <-- API'den gelen uyarı burada
    } catch (error) {
      console.error("Tahmin hatası:", error);
      setTahmin("Tahmin sırasında hata oluştu.");
    }
  };

  return (
    <div className="app-container">
      <h1>İlaç Kutusu Yazısı Tanıma</h1>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Tahmin Et</button>

      {previewUrl && (
        <img src={previewUrl} alt="Yüklenen Görsel" className="preview" />
      )}

      {/* Uyarı varsa göster */}
      {uyari && (
        <div className="uyari" style={{ marginTop: "15px", color: "#ff6600" }}>
          ⚠️ <strong>{uyari}</strong>
        </div>
      )}

      {/* Tahmin edilen yazı */}
      {tahmin && (
        <div className="tahmin-card" style={{ marginTop: "20px" }}>
          <h2>Tahmin Edilen Yazı:</h2>
          <p style={{ fontSize: "24px", fontWeight: "bold" }}>{tahmin}</p>
        </div>
      )}

      {/* CSV'den gelen ilaç bilgisi */}
      {ilacBilgisi && typeof ilacBilgisi === "object" && (
        <div className="tahmin-card" style={{ textAlign: "left", marginTop: "30px" }}>
          <h3>İlaç Bilgileri</h3>
          <ul>
            {Object.entries(ilacBilgisi).map(([key, value]) => (
              <li key={key}>
                <strong>{key}:</strong> {value}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
