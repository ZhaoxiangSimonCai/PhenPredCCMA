import os
from PhenPred.vae import plot_folder


runtime_files_folder = os.path.join(plot_folder, "files")
configs_folder = os.path.join(plot_folder, "configs")
config_history_folder = os.path.join(configs_folder, "history")
legacy_config_folder = os.path.join(plot_folder, "file")


def ensure_vae_artifact_dirs():
    for path in (runtime_files_folder, configs_folder, config_history_folder):
        os.makedirs(path, exist_ok=True)


def runtime_artifact_path(filename):
    return os.path.join(runtime_files_folder, filename)


def config_artifact_path(filename):
    return os.path.join(configs_folder, filename)


def config_history_artifact_path(filename):
    return os.path.join(config_history_folder, filename)


def default_hyperparameters_path():
    return config_artifact_path("hyperparameters.json")


def timestamped_hyperparameters_output_path(timestamp):
    return config_history_artifact_path(f"{timestamp}_hyperparameters.json")


def _first_existing_path(candidates):
    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = os.path.expanduser(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_existing_path(path, search_dirs=()):
    if path is None:
        return None

    path = os.path.expanduser(str(path))
    basename = os.path.basename(path)
    candidates = []

    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.extend([path, os.path.join(os.getcwd(), path)])

    for search_dir in search_dirs:
        if search_dir is None:
            continue
        if not os.path.isabs(path):
            candidates.append(os.path.join(search_dir, path))
        candidates.append(os.path.join(search_dir, basename))

    return _first_existing_path(candidates)


def resolve_runtime_artifact_path(path):
    return resolve_existing_path(path, search_dirs=(runtime_files_folder,))


def resolve_config_artifact_path(path):
    return resolve_existing_path(
        path,
        search_dirs=(
            configs_folder,
            config_history_folder,
            runtime_files_folder,
            legacy_config_folder,
        ),
    )


def resolve_timestamped_hyperparameters_path(timestamp):
    filename = f"{timestamp}_hyperparameters.json"
    return _first_existing_path(
        (
            config_history_artifact_path(filename),
            runtime_artifact_path(filename),
            os.path.join(legacy_config_folder, filename),
        )
    ) or config_history_artifact_path(filename)
