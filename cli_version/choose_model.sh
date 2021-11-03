echo "Choose one of the following models:"
unset options i
while IFS= read -r -d $'\0' f; do
  options[i++]="$f"
done < <(find /content/models/ -maxdepth 1 -type f -name "*.pytorch" -print0 )

select opt in "${options[@]}" "Stop the script"; do
  case $opt in
    *.pytorch)
      echo "Model $opt selected"
      
      path=$opt

      dir=${opt%/*}
      final_path="$dir/final-model.pytorch"

      cp -R $opt $final_path
      
      break
      ;;
    "Stop the script")
      echo "You chose to stop"
      break
      ;;
    *)
      echo "This is not a number"
      ;;
  esac
done