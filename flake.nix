{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
    poetry2nix = {
      url = "github:nix-community/poetry2nix?rev=3c92540611f42d3fb2d0d084a6c694cd6544b609";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix, pre-commit-hooks }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication overrides;
      in
      {
        checks = {
          pre-commit-check = pre-commit-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              ruff.enable = true;
            };
          };
        };
        packages = {
          softsensormonitorapp = mkPoetryApplication {
            projectDir = ./.;
            overrides = overrides.withDefaults (self: super: {
              pywavelets = pkgs.python311Packages.pywavelets;
              scikit-learn = pkgs.python311Packages.scikit-learn;
              scikit-image = pkgs.python311Packages.scikit-image;
            });
          };
          default = self.packages.${system}.softsensormonitorapp;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.softsensormonitorapp ];
          packages = with pkgs; [
            poetry
            ruff
            pre-commit
          ];
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };
      });
}
