use anyhow::{Context, Result};
use rcgen::{CertificateParams, DistinguishedName, KeyPair};
use std::path::Path;
use tonic::transport::{Certificate, Identity, ServerTlsConfig};
use tracing::info;

/// TLS certificate and key for this node, stored at `~/.config/noddle/tls/`.
/// On first run a self-signed cert is generated automatically.
/// The certificate's fingerprint is included in NodeCapability so peers can
/// verify the connection identity against the registry.
pub struct NodeTls {
    pub cert_pem: Vec<u8>,
    pub key_pem:  Vec<u8>,
}

impl NodeTls {
    /// Load existing cert/key or generate a new self-signed pair.
    pub fn load_or_generate(tls_dir: &Path) -> Result<Self> {
        let cert_path = tls_dir.join("node.crt");
        let key_path  = tls_dir.join("node.key");

        if cert_path.exists() && key_path.exists() {
            let cert_pem = std::fs::read(&cert_path)
                .context("reading node.crt")?;
            let key_pem = std::fs::read(&key_path)
                .context("reading node.key")?;
            info!("loaded existing TLS certificate");
            return Ok(Self { cert_pem, key_pem });
        }

        std::fs::create_dir_all(tls_dir)
            .context("creating TLS directory")?;

        let tls = Self::generate()
            .context("generating self-signed certificate")?;

        std::fs::write(&cert_path, &tls.cert_pem)
            .context("writing node.crt")?;
        std::fs::write(&key_path, &tls.key_pem)
            .context("writing node.key")?;

        // Restrict key file permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))
                .context("setting key file permissions")?;
        }

        info!("generated new self-signed TLS certificate");
        Ok(tls)
    }

    fn generate() -> Result<Self> {
        let mut params = CertificateParams::default();
        let mut dn = DistinguishedName::new();
        dn.push(rcgen::DnType::CommonName, "noddle-node");
        params.distinguished_name = dn;

        let key_pair = KeyPair::generate()?;
        let cert = params.self_signed(&key_pair)?;

        Ok(Self {
            cert_pem: cert.pem().into_bytes(),
            key_pem:  key_pair.serialize_pem().into_bytes(),
        })
    }

    /// Produce the tonic ServerTlsConfig for the gRPC server.
    pub fn server_tls_config(&self) -> Result<ServerTlsConfig> {
        let identity = Identity::from_pem(&self.cert_pem, &self.key_pem);
        Ok(ServerTlsConfig::new().identity(identity))
    }

    /// DER-encoded certificate for fingerprinting and peer verification.
    pub fn cert_der(&self) -> Result<Vec<u8>> {
        // Parse PEM and return the raw DER bytes
        let pem = std::str::from_utf8(&self.cert_pem).context("cert pem is not utf8")?;
        let der = pem
            .lines()
            .filter(|l| !l.starts_with("-----"))
            .collect::<String>();
        use std::io::Read;
        let mut decoder = base64::read::DecoderReader::new(
            std::io::Cursor::new(der.as_bytes()),
            &base64::engine::general_purpose::STANDARD,
        );
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).context("decoding cert DER")?;
        Ok(out)
    }

    /// Hex fingerprint (SHA-256) of the certificate — stored in NodeCapability.
    pub fn fingerprint(&self) -> Result<String> {
        use std::fmt::Write;
        let der = self.cert_der()?;
        // Simple SHA-256 via ring
        let digest = ring::digest::digest(&ring::digest::SHA256, &der);
        let mut hex = String::with_capacity(64);
        for byte in digest.as_ref() {
            write!(hex, "{:02x}", byte).unwrap();
        }
        Ok(hex)
    }

    pub fn client_ca(&self) -> Certificate {
        Certificate::from_pem(&self.cert_pem)
    }
}
