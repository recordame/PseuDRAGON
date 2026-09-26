/* Exercise the real browser method with an offline DOM/fetch boundary. */
const fs = require("fs");
const vm = require("vm");
const assert = require("assert/strict");
const root = process.argv[2];
const payloads = JSON.parse(fs.readFileSync(0, "utf8"));
const source = fs.readFileSync(`${root}/web_interface/static/js/app.js`, "utf8");

async function render(payload) {
    const cascade = { textContent: "" };
    const context = vm.createContext({
        document: { querySelector: () => cascade },
        window: { addEventListener: () => {} },
        fetch: async (url) => {
            assert.equal(url, "/api/runtime_config");
            return { ok: true, json: async () => payload };
        },
        console: { error: () => {} },
    });
    vm.runInContext(source + "\nglobalThis.TestApp = PseuDRAGONApp;", context);
    const app = Object.create(context.TestApp.prototype);
    app.elements = Object.fromEntries(
        [
            "runtimeModeBadge",
            "rankingMix",
            "authorityHierarchy",
            "evidenceIndex",
            "hitlRuleGate",
            "runtimeConfigNote",
        ].map((key) => [
            key,
            {
                textContent: "",
                previousElementSibling: { textContent: "" },
                classList: { toggle: () => {}, add: () => {} },
            },
        ])
    );
    await app.loadRuntimeConfig();
    return { elements: app.elements, cascade };
}

(async () => {
    let result = await render(payloads.source_order);
    assert.match(result.elements.rankingMix.textContent, /otherwise try the next source/);
    assert.equal(
        result.elements.authorityHierarchy.textContent,
        "Institutional policy → National law → International guidance"
    );
    assert.match(
        result.elements.runtimeConfigNote.textContent,
        /Expert review confirms applicability/
    );
    assert.doesNotMatch(result.elements.rankingMix.textContent, /α|NaN|authority|0\.7/);
    assert.match(result.cascade.textContent, /source-ordered retrieval/);
    assert.equal(
        result.elements.authorityHierarchy.previousElementSibling.textContent,
        "Search order"
    );

    result = await render(payloads.legacy);
    assert.match(result.elements.rankingMix.textContent, /Legacy · α=0\.7/);
    assert.match(result.elements.authorityHierarchy.textContent, /Institutional policy 0\.8/);
    assert.match(result.cascade.textContent, /weighted retrieval \(legacy\)/);

    result = await render(payloads.lexical);
    assert.equal(result.elements.rankingMix.textContent, "Lexical retrieval · source gap 0.20");
    assert.doesNotMatch(result.elements.authorityHierarchy.textContent, /0\.8|0\.6|→/);
    assert.equal(
        result.elements.authorityHierarchy.previousElementSibling.textContent,
        "Evidence sources"
    );

    const disabled = JSON.parse(JSON.stringify(payloads.source_order));
    disabled.features.rag_enabled = false;
    result = await render(disabled);
    assert.match(result.elements.runtimeModeBadge.textContent, /RAG OFF/);

    const unknown = JSON.parse(JSON.stringify(payloads.source_order));
    unknown.ranking.retrieval_mode = "unrecognized";
    result = await render(unknown);
    assert.equal(result.elements.runtimeModeBadge.textContent, "CONFIG ERROR");
    assert.equal(result.elements.rankingMix.textContent, "Unavailable");
    console.log("5 runtime rendering checks passed");
})().catch((error) => {
    console.error(error);
    process.exit(1);
});
