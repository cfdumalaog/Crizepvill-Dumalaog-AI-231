package com.dumalaog.me3.android;

import android.Manifest;
import android.app.Activity;
import android.database.Cursor;
import android.content.Intent;
import android.content.ContentResolver;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import android.provider.ContactsContract;

import org.json.JSONObject;

import java.util.Locale;

/**
 * Thin Android client for the ME3 Qwen Pi service.
 *
 * The Python voice loop, models, tools, and microphone remain on the Pi. The
 * phone is the operator console and browser PCM speaker, just like a LAN
 * browser, with a remembered editable endpoint rather than a hard-coded
 * deployment address.
 */
public final class MainActivity extends Activity {
    private static final int RECORD_AUDIO_REQUEST = 41;
    private static final int CALL_PHONE_REQUEST = 42;
    private static final int READ_CONTACTS_REQUEST = 43;
    private static final String PREFS = "me3_android";
    private static final String ENDPOINT_KEY = "endpoint";
    private static final String DEFAULT_ENDPOINT = "http://cfdfnjrpi5.local:8126/";
    private static final int INK = Color.rgb(226, 232, 240);
    private static final int MUTED = Color.rgb(148, 163, 184);
    private static final int PANEL = Color.rgb(11, 18, 32);

    private SharedPreferences preferences;
    private EditText endpointInput;
    private TextView connectStatus;
    private TextView toolbarStatus;
    private WebView webView;
    private PermissionRequest pendingAudioRequest;
    private String loadedHost;
    private String pendingCallContact;
    private String pendingCallActionId;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(PANEL);
        getWindow().setNavigationBarColor(PANEL);
        preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        String saved = preferences.getString(ENDPOINT_KEY, "");
        if (saved == null || saved.trim().isEmpty()) {
            showConnectionScreen(DEFAULT_ENDPOINT);
        } else {
            showWebClient(normalizeEndpoint(saved));
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private TextView label(String text, float size, int color) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(size);
        view.setTextColor(color);
        return view;
    }

    private LinearLayout.LayoutParams params(int width, int height) {
        return new LinearLayout.LayoutParams(width, height);
    }

    private void showConnectionScreen(String initialEndpoint) {
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(24), dp(28), dp(24), dp(24));
        content.setBackgroundColor(PANEL);

        TextView title = label("ME3 QWEN", 30, INK);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        content.addView(title, params(-1, -2));

        TextView subtitle = label("Android companion for the Raspberry Pi full-duplex agent", 15, MUTED);
        subtitle.setPadding(0, dp(6), 0, dp(26));
        content.addView(subtitle, params(-1, -2));

        TextView endpointLabel = label("ME3 server address", 13, MUTED);
        content.addView(endpointLabel, params(-1, -2));

        endpointInput = new EditText(this);
        endpointInput.setSingleLine(true);
        endpointInput.setText(initialEndpoint);
        endpointInput.setTextColor(INK);
        endpointInput.setHintTextColor(Color.rgb(100, 116, 139));
        endpointInput.setHint("http://192.168.x.x:8126/");
        endpointInput.setTextSize(16);
        endpointInput.setInputType(android.text.InputType.TYPE_CLASS_TEXT
                | android.text.InputType.TYPE_TEXT_VARIATION_URI);
        endpointInput.setSelectAllOnFocus(false);
        endpointInput.setPadding(dp(12), dp(10), dp(12), dp(10));
        content.addView(endpointInput, new LinearLayout.LayoutParams(-1, dp(52)));

        Button connect = new Button(this);
        connect.setText("Connect to ME3");
        connect.setTextColor(Color.WHITE);
        connect.setAllCaps(false);
        connect.setOnClickListener(v -> connectFromInput());
        LinearLayout.LayoutParams connectParams = params(-1, dp(52));
        connectParams.topMargin = dp(14);
        content.addView(connect, connectParams);

        connectStatus = label("", 13, Color.rgb(248, 191, 36));
        connectStatus.setPadding(0, dp(14), 0, dp(12));
        content.addView(connectStatus, params(-1, -2));

        TextView explanation = label(
                "The Pi keeps the Python voice loop, ASR/TTS services, tools, and microphone. "
                        + "This APK is the Android operator console and voice-output client. "
                        + "Put the phone and Pi on the same trusted LAN, then tap Connect browser speaker inside ME3.",
                14, MUTED);
        explanation.setLineSpacing(0f, 1.18f);
        content.addView(explanation, params(-1, -2));

        TextView permissions = label(
                "The app permits cleartext LAN HTTP because the current Pi service is HTTP, not public hosting. "
                        + "No API key or model credential is stored in the APK. Phone calls use the phone's SIM and request call/contacts permissions only when you ask ME3 to call.",
                12, Color.rgb(100, 116, 139));
        permissions.setPadding(0, dp(22), 0, 0);
        content.addView(permissions, params(-1, -2));

        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(PANEL);
        scroll.addView(content);
        setContentView(scroll);
    }

    private void connectFromInput() {
        try {
            String endpoint = normalizeEndpoint(endpointInput.getText().toString());
            preferences.edit().putString(ENDPOINT_KEY, endpoint).apply();
            showWebClient(endpoint);
        } catch (IllegalArgumentException error) {
            connectStatus.setText(error.getMessage());
        }
    }

    private String normalizeEndpoint(String raw) {
        String endpoint = raw == null ? "" : raw.trim();
        if (endpoint.isEmpty()) {
            throw new IllegalArgumentException("Enter the Pi ME3 HTTP address first.");
        }
        if (!endpoint.contains("://")) {
            endpoint = "http://" + endpoint;
        }
        Uri uri = Uri.parse(endpoint);
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        if (!(scheme.equals("http") || scheme.equals("https")) || uri.getHost() == null) {
            throw new IllegalArgumentException("Use an HTTP or HTTPS address such as http://192.168.x.x:8126/");
        }
        if (uri.getUserInfo() != null) {
            throw new IllegalArgumentException("Do not put credentials in the ME3 URL.");
        }
        return endpoint.endsWith("/") ? endpoint : endpoint + "/";
    }

    private void showWebClient(String endpoint) {
        Uri endpointUri = Uri.parse(endpoint);
        loadedHost = endpointUri.getHost();

        LinearLayout shell = new LinearLayout(this);
        shell.setOrientation(LinearLayout.VERTICAL);
        shell.setBackgroundColor(PANEL);

        LinearLayout toolbar = new LinearLayout(this);
        toolbar.setGravity(Gravity.CENTER_VERTICAL);
        toolbar.setPadding(dp(12), 0, dp(6), 0);
        toolbar.setBackgroundColor(PANEL);

        toolbarStatus = label("Connecting to ME3…", 13, MUTED);
        toolbarStatus.setSingleLine(true);
        toolbarStatus.setEllipsize(android.text.TextUtils.TruncateAt.END);
        toolbar.addView(toolbarStatus, new LinearLayout.LayoutParams(0, dp(48), 1f));

        Button change = new Button(this);
        change.setText("Change");
        change.setAllCaps(false);
        change.setTextColor(INK);
        change.setOnClickListener(v -> showConnectionScreen(endpoint));
        toolbar.addView(change, params(dp(90), dp(48)));
        shell.addView(toolbar, params(-1, dp(48)));

        webView = new WebView(this);
        configureWebView();
        shell.addView(webView, new LinearLayout.LayoutParams(-1, 0, 1f));
        setContentView(shell);
        webView.loadUrl(endpoint);
    }

    private void configureWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);

        webView.setBackgroundColor(Color.rgb(2, 6, 23));
        webView.setVerticalScrollBarEnabled(false);
        webView.addJavascriptInterface(new Me3Bridge(), "Me3Native");
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, android.graphics.Bitmap favicon) {
                setToolbarStatus("Connecting to " + loadedHost + "…", MUTED);
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                setToolbarStatus("ME3 connected", Color.rgb(74, 222, 128));
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame()) {
                    setToolbarStatus("ME3 offline - check LAN/server", Color.rgb(248, 113, 113));
                    Toast.makeText(MainActivity.this, "Could not reach the ME3 server", Toast.LENGTH_LONG).show();
                }
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                if (loadedHost != null && loadedHost.equalsIgnoreCase(uri.getHost())
                        && ("http".equals(uri.getScheme()) || "https".equals(uri.getScheme()))) {
                    return false;
                }
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, uri));
                } catch (Exception ignored) {
                    Toast.makeText(MainActivity.this, "No app can open that link", Toast.LENGTH_SHORT).show();
                }
                return true;
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> handleWebPermission(request));
            }
        });
    }

    private void handleWebPermission(PermissionRequest request) {
        boolean wantsAudio = false;
        for (String resource : request.getResources()) {
            if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(resource)) {
                wantsAudio = true;
                break;
            }
        }
        if (!wantsAudio) {
            request.deny();
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            request.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
        } else {
            pendingAudioRequest = request;
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, RECORD_AUDIO_REQUEST);
        }
    }

    private void setToolbarStatus(String message, int color) {
        if (toolbarStatus != null) {
            toolbarStatus.setText(message);
            toolbarStatus.setTextColor(color);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CALL_PHONE_REQUEST) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                continueRequestedCall();
            } else {
                finishCall(false, "", "call_permission_denied");
            }
            return;
        }
        if (requestCode == READ_CONTACTS_REQUEST) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                continueRequestedCall();
            } else {
                finishCall(false, "", "contacts_permission_denied");
            }
            return;
        }
        if (requestCode != RECORD_AUDIO_REQUEST) {
            return;
        }
        if (pendingAudioRequest != null) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                pendingAudioRequest.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
            } else {
                pendingAudioRequest.deny();
                Toast.makeText(this, "Microphone permission was denied", Toast.LENGTH_SHORT).show();
            }
            pendingAudioRequest = null;
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        if (webView != null) {
            webView.stopLoading();
            webView.setWebChromeClient(null);
            webView.setWebViewClient(null);
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }

    private final class Me3Bridge {
        @JavascriptInterface
        public void placeCall(String contact, String actionId) {
            String safeContact = contact == null ? "" : contact.trim();
            String safeId = actionId == null ? "" : actionId.trim();
            runOnUiThread(() -> beginRequestedCall(safeContact, safeId));
        }
    }

    private void beginRequestedCall(String contact, String actionId) {
        if (contact.isEmpty() || actionId.isEmpty()) {
            reportDeviceAction(actionId, false, "", "call_contact_or_action_id_missing");
            return;
        }
        pendingCallContact = contact;
        pendingCallActionId = actionId;
        continueRequestedCall();
    }

    private void continueRequestedCall() {
        if (pendingCallActionId == null) return;
        if (checkSelfPermission(Manifest.permission.CALL_PHONE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CALL_PHONE}, CALL_PHONE_REQUEST);
            return;
        }
        String number = normalizePhoneNumber(pendingCallContact);
        if (number == null) {
            if (checkSelfPermission(Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.READ_CONTACTS}, READ_CONTACTS_REQUEST);
                return;
            }
            number = resolveContactNumber(pendingCallContact);
            if (number == null) {
                finishCall(false, "", "contact_not_found_or_ambiguous");
                return;
            }
        }
        try {
            Intent intent = new Intent(Intent.ACTION_CALL, Uri.parse("tel:" + Uri.encode(number)));
            startActivity(intent);
            finishCall(true, "call_started_by_android_phone", "");
        } catch (SecurityException error) {
            finishCall(false, "", "call_permission_denied");
        } catch (Exception error) {
            finishCall(false, "", "phone_call_unavailable");
        }
    }

    private String normalizePhoneNumber(String value) {
        String input = value == null ? "" : value.trim();
        StringBuilder digits = new StringBuilder();
        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (Character.isDigit(c) || (c == '+' && digits.length() == 0)) digits.append(c);
        }
        String number = digits.toString();
        return number.length() >= 7 ? number : null;
    }

    private String resolveContactNumber(String name) {
        ContentResolver resolver = getContentResolver();
        Uri uri = Uri.withAppendedPath(ContactsContract.CommonDataKinds.Phone.CONTENT_FILTER_URI,
                Uri.encode(name));
        Cursor cursor = null;
        try {
            cursor = resolver.query(uri,
                    new String[]{ContactsContract.CommonDataKinds.Phone.NUMBER},
                    null, null, null);
            if (cursor == null) return null;
            String match = null;
            while (cursor.moveToNext()) {
                String candidate = cursor.getString(0);
                if (normalizePhoneNumber(candidate) == null) continue;
                if (match != null && !match.equals(candidate)) return null;
                match = candidate;
            }
            return match == null ? null : normalizePhoneNumber(match);
        } catch (SecurityException error) {
            return null;
        } finally {
            if (cursor != null) cursor.close();
        }
    }

    private void finishCall(boolean ok, String status, String error) {
        String actionId = pendingCallActionId;
        pendingCallActionId = null;
        pendingCallContact = null;
        reportDeviceAction(actionId, ok, status, error);
    }

    private void reportDeviceAction(String actionId, boolean ok, String status, String error) {
        if (webView == null) return;
        String js = "window.me3CompleteDeviceAction(" + JSONObject.quote(actionId == null ? "" : actionId)
                + "," + ok + "," + JSONObject.quote(status == null ? "" : status)
                + "," + JSONObject.quote(error == null ? "" : error) + ");";
        runOnUiThread(() -> {
            if (webView != null) webView.evaluateJavascript(js, null);
        });
    }
}
