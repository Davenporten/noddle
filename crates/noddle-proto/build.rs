fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        // Add serde derives to every generated message type so the registry
        // can serialize/deserialize NodeCapability to/from JSON.
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .compile_protos(&["../../proto/noddle.proto"], &["../../proto"])?;
    Ok(())
}
