module License

// Key format: CONF-<base64url(payload_json)>.<base64url(signature_64bytes)>
// Payload: JSON { sub, exp, tier, kid }
// Signature: Ed25519 over raw payload bytes (not the base64url encoding)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type LicensePayload = {
    sub:  string   // email
    exp:  int64    // Unix timestamp (0 = perpetual)
    tier: string   // "pro"
    kid:  string   // "k1"
}

type LicenseStatus =
    | Valid       of LicensePayload
    | Expired     of LicensePayload   // expired > 14 days ago
    | GracePeriod of LicensePayload   // expired within last 14 days
    | Invalid     of string           // reason

// ---------------------------------------------------------------------------
// Embedded keys (production key pair, generated 2026-03-04)
// Private key is #if !FABLE_COMPILER only -- never compiled into JS output.
// ---------------------------------------------------------------------------

let private PUBLIC_KEY_BYTES : byte[] =
    [| 0x22uy; 0x9duy; 0x21uy; 0x74uy; 0x07uy; 0x84uy; 0xd3uy; 0xceuy
       0xd5uy; 0x9cuy; 0x91uy; 0xebuy; 0xefuy; 0x2duy; 0x6duy; 0xdcuy
       0xc1uy; 0x66uy; 0x5cuy; 0xcauy; 0xaduy; 0x39uy; 0xeduy; 0xbauy
       0x97uy; 0xdcuy; 0x89uy; 0xfauy; 0x65uy; 0x43uy; 0xd2uy; 0x57uy |]

#if !FABLE_COMPILER
let private PRIVATE_KEY_SEED : byte[] =
    [| 0x71uy; 0x49uy; 0xacuy; 0x35uy; 0x2cuy; 0x37uy; 0xc2uy; 0x9auy
       0xa9uy; 0xb1uy; 0x10uy; 0x5cuy; 0x21uy; 0x04uy; 0xfbuy; 0xdduy
       0xa2uy; 0xb1uy; 0x26uy; 0xa0uy; 0x6duy; 0x1buy; 0x25uy; 0x47uy
       0x20uy; 0x4duy; 0xa2uy; 0x15uy; 0x56uy; 0x14uy; 0x4auy; 0xd2uy |]
#endif

// ---------------------------------------------------------------------------
// Base64url helpers (no padding, - for +, _ for /)
// ---------------------------------------------------------------------------

let private toBase64Url (bytes: byte[]) : string =
    System.Convert.ToBase64String(bytes)
        .TrimEnd('=')
        .Replace('+', '-')
        .Replace('/', '_')

let private fromBase64Url (s: string) : Result<byte[], string> =
    try
        let padded =
            let rem = s.Length % 4
            if rem = 0 then s
            elif rem = 2 then s + "=="
            elif rem = 3 then s + "="
            else s + "==="   // rem=1 is invalid but let Convert catch it
        let standard = padded.Replace('-', '+').Replace('_', '/')
        Ok (System.Convert.FromBase64String(standard))
    with ex ->
        Error ("base64url decode failed: " + ex.Message)

// ---------------------------------------------------------------------------
// Minimal JSON parser for LicensePayload
// Handles only the specific shape: {"sub":..., "exp":..., "tier":..., "kid":...}
// ---------------------------------------------------------------------------

let private extractJsonField (json: string) (field: string) : string option =
    // Find  "field": <value>  and return the raw value string (trimmed)
    let key = "\"" + field + "\":"
    let idx = json.IndexOf(key)
    if idx < 0 then None
    else
        let after = json.Substring(idx + key.Length).TrimStart()
        if after.StartsWith("\"") then
            // String value: find closing quote (not escaped)
            let mutable i = 1
            let mutable found = false
            while i < after.Length && not found do
                if after.[i] = '"' && (i = 0 || after.[i-1] <> '\\') then
                    found <- true
                else
                    i <- i + 1
            if found then Some (after.Substring(1, i - 1))
            else None
        else
            // Numeric value: read until comma, }, or whitespace
            let endIdx =
                after |> Seq.tryFindIndex (fun c -> c = ',' || c = '}' || System.Char.IsWhiteSpace c)
                |> Option.defaultValue after.Length
            Some (after.Substring(0, endIdx))

let private decodePayload (payloadBytes: byte[]) : Result<LicensePayload, string> =
    try
        let json = System.Text.Encoding.UTF8.GetString(payloadBytes)
        match extractJsonField json "sub", extractJsonField json "exp",
              extractJsonField json "tier", extractJsonField json "kid" with
        | Some sub, Some expStr, Some tier, Some kid ->
            match System.Int64.TryParse(expStr) with
            | true, exp ->
                Ok { sub = sub; exp = exp; tier = tier; kid = kid }
            | _ ->
                Error ("invalid exp value: " + expStr)
        | _ ->
            Error "missing required field in payload JSON"
    with ex ->
        Error ("payload decode failed: " + ex.Message)

// ---------------------------------------------------------------------------
// Ed25519 signature verification
// ---------------------------------------------------------------------------

let private verifySignature (payloadBytes: byte[]) (signatureBytes: byte[]) : bool =
#if FABLE_COMPILER
    // Fable path: the TS host performs verification; always return true here.
    // The F# Fable module is only used for payload decoding.
    true
#else
    if signatureBytes.Length <> 64 then false
    else
        try
            let algo = NSec.Cryptography.Ed25519.Ed25519
            match NSec.Cryptography.PublicKey.TryImport(algo, PUBLIC_KEY_BYTES, NSec.Cryptography.KeyBlobFormat.RawPublicKey) with
            | true, pubKey ->
                algo.Verify(pubKey, System.ReadOnlySpan<byte>(payloadBytes),
                            System.ReadOnlySpan<byte>(signatureBytes))
            | _ -> false
        with _ -> false
#endif

// ---------------------------------------------------------------------------
// Key parsing
// ---------------------------------------------------------------------------

let private parseKey (keyStr: string) : Result<byte[] * byte[], string> =
    if not (keyStr.StartsWith("CONF-")) then
        Error "key must start with CONF-"
    else
        let rest = keyStr.Substring(5)
        let dot = rest.IndexOf('.')
        if dot < 0 then
            Error "key missing '.' separator"
        else
            let payloadPart = rest.Substring(0, dot)
            let sigPart = rest.Substring(dot + 1)
            match fromBase64Url payloadPart with
            | Error e -> Error ("payload decode: " + e)
            | Ok payloadBytes ->
                match fromBase64Url sigPart with
                | Error e -> Error ("signature decode: " + e)
                | Ok sigBytes -> Ok (payloadBytes, sigBytes)

// ---------------------------------------------------------------------------
// Grace period / expiry logic
// ---------------------------------------------------------------------------

let private GRACE_SECONDS = 14L * 86400L

let private checkExpiry (payload: LicensePayload) : LicenseStatus =
    if payload.exp = 0L then
        Valid payload   // perpetual
    else
        let now = System.DateTimeOffset.UtcNow.ToUnixTimeSeconds()
        if now <= payload.exp then
            Valid payload
        elif now <= payload.exp + GRACE_SECONDS then
            GracePeriod payload
        else
            Expired payload

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Validate a license key string. Returns Valid, GracePeriod, Expired, or Invalid.
let validateLicense (keyStr: string) : LicenseStatus =
    match parseKey keyStr with
    | Error reason -> Invalid reason
    | Ok (payloadBytes, sigBytes) ->
        if not (verifySignature payloadBytes sigBytes) then
            Invalid "signature verification failed"
        else
            match decodePayload payloadBytes with
            | Error reason -> Invalid reason
            | Ok payload ->
                if payload.tier <> "pro" then
                    Invalid ("unknown tier: " + payload.tier)
                else
                    checkExpiry payload

/// Generate a signed license key. Only available in .NET (not Fable).
/// Used by the --generate-key CLI command.
#if !FABLE_COMPILER
let generateKey (email: string) (expUnix: int64) (tier: string) : string =
    let payload =
        sprintf "{\"sub\":\"%s\",\"exp\":%d,\"tier\":\"%s\",\"kid\":\"k1\"}"
            email expUnix tier
    let payloadBytes = System.Text.Encoding.UTF8.GetBytes(payload)

    let algo = NSec.Cryptography.Ed25519.Ed25519
    // Import the private key seed for signing only (no re-export needed)
    let privKey = NSec.Cryptography.Key.Import(algo, System.ReadOnlySpan<byte>(PRIVATE_KEY_SEED),
                      NSec.Cryptography.KeyBlobFormat.RawPrivateKey)
    let sigBytes = algo.Sign(privKey, System.ReadOnlySpan<byte>(payloadBytes))

    "CONF-" + toBase64Url payloadBytes + "." + toBase64Url sigBytes
#endif
