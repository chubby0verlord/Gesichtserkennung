# Dokumentation zum Python-Code zur Gesichtserkennung und -clusterung

## Einleitung

Der vorliegende Python-Code nutzt verschiedene Bibliotheken zur Erkennung, Speicherung und Clusterung von Gesichtern. Dabei werden Gesichter mittels Webcam erfasst, mit einem Referenzbild verglichen und in einem Verzeichnis gespeichert. Anschließend werden die gespeicherten Gesichter kodiert und geclustert.

## Benutzte Bibliotheken

- `threading`
- `cv2` (OpenCV)
- `os`
- `DeepFace`
- `pickle`
- `Path` (pathlib)
- `imutils`
- `face_recognition`
- `DBSCAN` (sklearn.cluster)
- `numpy`
- `uuid`
- `playsound`

## Funktionsweise

### Gesichtserkennung und Speicherung

1. **Variableninitialisierung**: Initialisierung der Zähler und Verzeichnisse.
2. **Laden des Gesichtserkennungsmodells**: Mittels OpenCV.
3. **Webcam-Setup**: Initialisierung der Webcam zur Videoaufnahme.
4. **Frame-Prüfung**: Vergleich jedes 30. Frames mit einem Referenzbild.
5. **Bounding-Box-Zeichnung**: Erkennung und Markierung von Gesichtern im Frame.
6. **Speicherung der Gesichter**: Speichern erkannter Gesichter in einem Verzeichnis.
7. **Audio-Benachrichtigung**: Abspielen eines Alarms bei Übereinstimmung mit dem Referenzbild.

### Kodierung der Gesichter

1. **Bilderpfade laden**: Laden der Pfade zu den gespeicherten Gesichtsbildern.
2. **Gesichter kodieren**: Erstellen von Gesichtsmerkmal-Vektoren mittels `face_recognition`.
3. **Speichern der Kodierungen**: Speichern der kodierten Gesichter in einer Datei (`pickle`).

### Clusterung der Gesichter

1. **Laden der Kodierungen**: Laden der gespeicherten Gesichtskodierungen.
2. **Clusterbildung**: Clusterung der Gesichter mittels DBSCAN-Algorithmus.
3. **Speichern der Cluster**: Speichern und Visualisierung der geclusterten Gesichter.

## Hauptfunktionen

### `check_frame(frame)`
- Prüft, ob das übergebene Frame dem Referenzbild entspricht.
- Nutzt `DeepFace.verify()` für den Vergleich.

### `draw_detection_box(faces, frame)`
- Zeichnet Bounding-Boxen um erkannte Gesichter im Frame.
- Ruft `save_face(face_img)` auf, um erkannte Gesichter zu speichern.

### `save_face(face_img)`
- Speichert erkannte Gesichter in einem definierten Verzeichnis.

### `play_sound(sound_file)`
- Spielt einen Alarmton ab.

### `video_detection()`
- Hauptschleife zur Gesichtserkennung und -anzeige mittels Webcam.
- Prüft Frames regelmäßig und speichert erkannte Gesichter.

### `encode_faces()`
- Kodiert die gespeicherten Gesichter und speichert die Kodierungen.

### `cluster_faces()`
- Führt die Clusterung der kodierten Gesichter durch.
- Speichert und visualisiert die Ergebnisse.

## Ausführung

- Startet die Hauptfunktion `video_detection()`, kodiert anschließend die Gesichter und clustert diese.

```python
if __name__ == "__main__":
    video_detection()
    encode_faces()
    cluster_faces()