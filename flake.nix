{
  description = "Development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (ps: with ps; [
        numpy
        pillow
      ]);
    in {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [ mpi cmake gnumake python clang-tools ];
        buildInputs = with pkgs; [ mpi ];
      };
    });
}
