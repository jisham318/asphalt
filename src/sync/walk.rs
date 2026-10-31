use crate::{
    asset::{self, Asset, AssetKind},
    cli::SyncTarget,
    config::Config,
    hash::Hash,
    lockfile::Lockfile,
    sync::TargetBackend,
};
use anyhow::Context;
use fs_err::tokio as fs;
use log::{debug, warn};
use relative_path::PathExt;
use resvg::usvg::fontdb;
use std::{
    collections::{
        HashMap,
        hash_map::{self},
    },
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    sync::{Mutex, Semaphore, mpsc::UnboundedSender},
    task::JoinSet,
};
use walkdir::WalkDir;

pub struct Params {
    pub target: SyncTarget,
    pub only: Option<Vec<AssetKind>>,
    pub existing_lockfile: Lockfile,
    pub font_db: Arc<fontdb::Database>,
    pub backend: Option<TargetBackend>,
}

struct InputState {
    params: Arc<Params>,
    input_name: String,
    input_prefix: PathBuf,
    seen_hashes: Arc<Mutex<HashMap<Hash, PathBuf>>>,
    bleed: bool,
    convert_meshes: bool,
}

pub async fn walk(params: Params, config: &Config, tx: &UnboundedSender<super::Event>) {
    let params = Arc::new(params);

    for (input_name, input) in &config.inputs {
        let input_prefix = config.project_dir.join(input.include.get_prefix());
        let convert_meshes = input.studio_meshes && matches!(params.target, SyncTarget::Studio);

        let state = Arc::new(InputState {
            params: params.clone(),
            input_name: input_name.clone(),
            input_prefix: input_prefix.clone(),
            seen_hashes: Arc::new(Mutex::new(HashMap::new())),
            bleed: input.bleed,
            convert_meshes,
        });

        let mut join_set = JoinSet::new();
        let semaphore = Arc::new(Semaphore::new(50));

        for entry in WalkDir::new(&input_prefix)
            .into_iter()
            .filter_entry(|entry| {
                let path = entry.path();
                if path == input_prefix {
                    return true;
                }
                if entry.file_type().is_dir() {
                    return true;
                }
                if let Ok(rel_path) = path.strip_prefix(&config.project_dir) {
                    input.include.is_match(rel_path)
                } else {
                    false
                }
            })
        {
            let Ok(entry) = entry else { continue };

            let path = entry.into_path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path.extension() else {
                continue;
            };
            if !asset::is_supported_extension(ext) {
                continue;
            }
            // OBJ files can't be uploaded, so they're only synced as converted meshes
            if ext == "obj" && !convert_meshes {
                continue;
            }

            let state = state.clone();
            let semaphore = semaphore.clone();
            let tx = tx.clone();

            tx.send(super::Event::Discovered(path.clone())).unwrap();

            join_set.spawn(async move {
                let _permit = semaphore.acquire_owned().await.unwrap();

                tx.send(super::Event::InFlight(path.clone())).unwrap();

                if let Err(e) = process_entry(state.clone(), &path, &tx).await {
                    warn!("Failed to process file {}: {e:?}", path.display());
                    tx.send(super::Event::Failed(path.clone())).unwrap();
                }
            });
        }

        while join_set.join_next().await.is_some() {}
    }
}

async fn process_entry(
    state: Arc<InputState>,
    path: &Path,
    tx: &UnboundedSender<super::Event>,
) -> anyhow::Result<()> {
    debug!("Handling entry: {}", path.display());

    let rel_path = path.relative_to(&state.input_prefix)?;

    let data = fs::read(path).await?;

    let mut asset = Asset::new(rel_path.clone(), data.into()).context("Failed to create asset")?;

    let lockfile_entry = state
        .params
        .existing_lockfile
        .get(&state.input_name, &asset.hash);

    {
        let mut seen_hashes = state.seen_hashes.lock().await;

        match seen_hashes.entry(asset.hash) {
            hash_map::Entry::Occupied(entry) => {
                let seen_path = entry.get();
                let rel_seen_path = seen_path.relative_to(&state.input_prefix)?;

                debug!("Duplicate asset found: {} -> {}", rel_path, rel_seen_path);

                let event = super::Event::Finished {
                    state: super::EventState::Duplicate,
                    input_name: state.input_name.clone(),
                    path: path.into(),
                    rel_path: rel_path.clone(),
                    asset_ref: lockfile_entry.map(Into::into),
                    hash: asset.hash,
                };
                tx.send(event).unwrap();

                return Ok(());
            }
            hash_map::Entry::Vacant(_) => {
                seen_hashes.insert(asset.hash, path.into());
            }
        }
    }

    if let Some(only) = &state.params.only
        && !only.contains(&asset.ty.kind())
    {
        let event = match lockfile_entry {
            Some(entry) => super::Event::Finished {
                state: super::EventState::Synced { new: false },
                input_name: state.input_name.clone(),
                path: path.into(),
                rel_path: asset.path.clone(),
                hash: asset.hash,
                asset_ref: Some(entry.into()),
            },
            None => {
                debug!("Skipping {} (type not in --only)", rel_path);
                super::Event::Skipped(path.into())
            }
        };
        tx.send(event).unwrap();
        return Ok(());
    }

    let always_target = matches!(state.params.target, SyncTarget::Studio | SyncTarget::Debug);
    let is_new = always_target || lockfile_entry.is_none();

    if is_new {
        let font_db = state.params.font_db.clone();
        let bleed = state.bleed;
        let convert_meshes = state.convert_meshes;

        asset = tokio::task::spawn_blocking(move || -> anyhow::Result<Asset> {
            let mut asset = asset;
            asset.process(font_db, bleed, convert_meshes)?;
            Ok(asset)
        })
        .await?
        .context("Failed to process asset")?;
    }

    let asset_ref = match state.params.backend {
        Some(ref backend) => backend.sync(&asset, lockfile_entry).await?,
        None => lockfile_entry.map(Into::into),
    };

    let event = super::Event::Finished {
        state: super::EventState::Synced { new: is_new },
        input_name: state.input_name.clone(),
        path: path.into(),
        rel_path: asset.path.clone(),
        hash: asset.hash,
        asset_ref,
    };
    tx.send(event).unwrap();

    Ok(())
}
