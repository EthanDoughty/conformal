import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// Embedded Ed25519 public key (matches src/License.fs PUBLIC_KEY_BYTES)
const PUBLIC_KEY_PEM = `-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAIp0hdAeE087VnJHr7y1t3MFmXMqtOe26l9yJ+mVD0lc=
-----END PUBLIC KEY-----`;

export interface LicensePayload {
    sub: string;
    exp: number;
    tier: string;
    kid: string;
}

export type LicenseStatus =
    | { kind: 'valid'; payload: LicensePayload }
    | { kind: 'grace'; payload: LicensePayload; daysLeft: number }
    | { kind: 'expired'; payload: LicensePayload }
    | { kind: 'invalid'; reason: string }
    | { kind: 'none' };

const GRACE_SECONDS = 14 * 86400;

export function validateLicense(keyStr: string): LicenseStatus {
    if (!keyStr) return { kind: 'none' };
    if (!keyStr.startsWith('CONF-')) {
        return { kind: 'invalid', reason: 'invalid format' };
    }

    const withoutPrefix = keyStr.substring(5);
    const dotIdx = withoutPrefix.indexOf('.');
    if (dotIdx < 0) return { kind: 'invalid', reason: 'missing signature' };

    const payloadB64 = withoutPrefix.substring(0, dotIdx);
    const sigB64 = withoutPrefix.substring(dotIdx + 1);

    let payloadBuf: Buffer;
    let sigBuf: Buffer;
    try {
        payloadBuf = Buffer.from(payloadB64, 'base64url');
        sigBuf = Buffer.from(sigB64, 'base64url');
    } catch {
        return { kind: 'invalid', reason: 'base64 decode failed' };
    }

    // Verify Ed25519 signature
    try {
        const keyObj = crypto.createPublicKey(PUBLIC_KEY_PEM);
        const valid = crypto.verify(null, payloadBuf, keyObj, sigBuf);
        if (!valid) return { kind: 'invalid', reason: 'bad signature' };
    } catch {
        return { kind: 'invalid', reason: 'signature verification error' };
    }

    let payload: LicensePayload;
    try {
        payload = JSON.parse(payloadBuf.toString('utf-8'));
    } catch {
        return { kind: 'invalid', reason: 'payload parse error' };
    }

    if (payload.tier !== 'pro') {
        return { kind: 'invalid', reason: `unknown tier: ${payload.tier}` };
    }

    // Check expiry
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp === 0) return { kind: 'valid', payload };
    if (now <= payload.exp) return { kind: 'valid', payload };
    const graceEnd = payload.exp + GRACE_SECONDS;
    if (now <= graceEnd) {
        const daysLeft = Math.ceil((graceEnd - now) / 86400);
        return { kind: 'grace', payload, daysLeft };
    }
    return { kind: 'expired', payload };
}

export function readLicenseFromFile(): string {
    try {
        const keyPath = path.join(os.homedir(), '.conformal', 'license.key');
        if (fs.existsSync(keyPath)) {
            return fs.readFileSync(keyPath, 'utf-8').trim();
        }
    } catch { /* ignore */ }
    return '';
}
