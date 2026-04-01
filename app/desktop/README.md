# EmoResonance Windows App

This folder contains the migrated desktop client source used as the UI base for the `lunwen` research repository.

## Current role

- Native login, activation, psychometric assessment, and owner binding
- OpenClaw desktop conversation entry
- Desktop-local speech transcription
- Device monitoring, settings sync, and remote robot control
- Emotion and proactive care visualization

## Dev run

1. Install dependencies
   ```bash
   npm install
   ```
2. Start the frontend
   ```bash
   npm run dev
   ```
3. Default frontend port is `3001`

## Research-repo migration note

This copy intentionally keeps source code only:

- included: UI source, assets, Electron entrypoints, services, and build config
- excluded: historical `release-*` outputs, `dist`, `node_modules`, and bundled vendor runtimes

The desktop app is treated as a product-facing shell for the proactive care research engine. Packaging and full runtime bundling will be reattached in the clean repo rather than copied wholesale from older workspaces.

## Backend contract

The desktop app is designed to work with the local backend launched by Electron, or with a remote backend that implements the same contract.

Important endpoints:

- `POST /api/auth/login`
- `GET /api/activation/state`
- `POST /api/assistant/send`
- `GET /api/assistant/session/status`
- `GET /api/desktop/runtime/status`
- `GET /api/desktop/voice/status`
- `POST /api/desktop/voice/transcribe`
- `GET /api/emotion/history`
- `GET /api/emotion/realtime`
- `GET /api/device/status`
- `GET /api/device/settings`
- `POST /api/device/settings`
- WebSocket `ws://<host>:8000/ws/events`

## First-time device onboarding

The old ESP BLE / SoftAP provisioning flow is no longer the active path.

The current provisioning component targets the Pi local onboarding API:

- `GET http://<pi-host>:8090/onboarding/state`
- `GET http://<pi-host>:8090/onboarding/networks`
- `POST http://<pi-host>:8090/onboarding/wifi`

Typical onboarding host while connected to the robot hotspot:

- `192.168.4.1:8090`

After Wi-Fi onboarding succeeds, the rest of setup continues in the native desktop activation flow:

1. Login
2. Identity confirmation
3. 8-dimension assessment
4. Owner face binding
