// secrets.example.h -- copy this file to secrets.h (same folder) and fill in your values.
// secrets.h is listed in .gitignore so your WiFi password is never uploaded.
#pragma once
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Local backend (laptop): its IP on the WiFi network, port 8000, no API key needed
const char* SERVER_URL    = "http://192.168.1.100:8000/readings";
#define API_KEY ""

// Online backend (Render): use these two lines instead, with the same key as the
// API_KEY setting on Render
// const char* SERVER_URL = "https://YOUR-BACKEND.onrender.com/readings";
// #define API_KEY "YOUR_SECRET_KEY"
