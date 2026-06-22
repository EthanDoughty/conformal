-- Neovim LSP client for the Conformal MATLAB shape analyzer.
--
-- The analyzer ships a Language Server over stdio, started with `conformal --lsp`.
-- This module registers it with no plugin dependencies using the built-in client,
-- so it works on a stock Neovim. Require it once from your config:
--
--     require("conformal").setup()
--
-- You get diagnostics, hover, document symbols, go-to-definition, quick-fix code
-- actions, and inferred-shape inlay hints. The `conformal` binary must be on $PATH,
-- or pass an absolute path through opts.cmd. Inlay hints need Neovim 0.10 or newer;
-- everything else works on 0.8 or newer.

local M = {}

-- A bare `.m` extension is ambiguous across MATLAB, Objective-C, and Mathematica.
-- Pin it to the matlab filetype so the client attaches; switch to "octave" if that
-- is your dialect.
local function ensure_filetype()
  vim.filetype.add({ extension = { m = "matlab" } })
end

-- Mirror the CLI's project resolution: walk upward for a .conformal.json config,
-- then a .git boundary, falling back to the file's own directory for loose files.
local function root_dir(fname)
  local found = vim.fs.find({ ".conformal.json", ".git" }, {
    path = vim.fs.dirname(fname),
    upward = true,
  })[1]
  return found and vim.fs.dirname(found) or vim.fs.dirname(fname)
end

-- opts:
--   cmd          command table, defaults to { "conformal", "--lsp" }
--   inlay_hints  enable inferred-shape inlay hints on attach, default true
function M.setup(opts)
  opts = opts or {}
  local cmd = opts.cmd or { "conformal", "--lsp" }
  local want_hints = opts.inlay_hints ~= false

  ensure_filetype()

  vim.api.nvim_create_autocmd("FileType", {
    pattern = { "matlab", "octave" },
    desc = "Start the Conformal language server",
    callback = function(args)
      -- vim.lsp.start reuses a client when name, cmd, and root_dir match, so
      -- reopening files in the same project shares one server process.
      vim.lsp.start({
        name = "conformal",
        cmd = cmd,
        root_dir = root_dir(vim.api.nvim_buf_get_name(args.buf)),
      })
    end,
  })

  if want_hints then
    vim.api.nvim_create_autocmd("LspAttach", {
      desc = "Turn on Conformal inlay hints",
      callback = function(args)
        local client = vim.lsp.get_client_by_id(args.data.client_id)
        if client and client.name == "conformal" and vim.lsp.inlay_hint then
          -- The enable signature settled in 0.10.1; pcall keeps older builds quiet.
          pcall(vim.lsp.inlay_hint.enable, true, { bufnr = args.buf })
        end
      end,
    })
  end
end

return M
