# Frontend

TensorFlow's own Embedding Projector, pointed at this repo's output.

    ./serve.sh          # then open http://localhost:8133/

`index.html` is the upstream standalone build from
`tensorflow/embedding-projector-standalone`, with two lines changed:

  * `config-json-path` points at `data/projector_config.json` instead of the
    97 MB demo dataset that is not in this repo.
  * the Bookmarks panel is hidden. Hidden, not deleted — the projector does
    `bookmarkPanel = this.$["bookmark-panel"]` and then calls `.initialize()`
    on it, so removing the element stops the whole app from starting.

`index.html.upstream` is the untouched original, so the diff is always visible:

    diff <(tr '>' '\n' < index.html.upstream) <(tr '>' '\n' < index.html)

## Where the data comes from

`data/` is a symlink into `../output/<run>`, not a copy. Re-run the pipeline and
this page picks it up on reload.

    ln -sfn ../output/<run> data      # to view a different run

Runs are named after the settings that produced them
(`nvidia-nemotron-3-embed-1b-free--d50-nn15-mcs5-ms3-eom`), so two settings can
sit side by side and a directory name always says what made it.

## Reading it

`cluster_name` leads the metadata, so points are labelled by concept. The other
columns — `title`, `cluster_id`, `size`, `month`, `document`, `url`, `text` —
are all selectable under "Label by" and "Color by".

**Turn "Spherize data" off.** It is on by default and it centres then
normalises every vector, which pushes all of them out to radius 1 and empties
the middle. Measured on this data: 84% of points land in the outer shell after
spherizing, against 11% before. The sphere is the checkbox, not the concepts.

Nothing on screen is a measurement. The view is 50 dimensions projected to 3,
and those 50 are themselves a projection of 2,048. Use it to check whether the
grouping looks sane, never to measure.
