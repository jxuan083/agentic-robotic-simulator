"""Reserve Alt + mouse for viewport navigation in the browser desktop."""

from pathlib import Path
import xml.etree.ElementTree as ET


def configure(source: Path, destination: Path) -> None:
    namespace = "http://openbox.org/3.4/rc"
    ET.register_namespace("", namespace)
    tree = ET.parse(source)
    # Frame bindings apply over the client area too. Openbox otherwise grabs
    # Alt + left/right/middle before Kit can orbit, dolly, or pan the camera.
    for context in tree.getroot().iter(f"{{{namespace}}}context"):
        if context.get("name") != "Frame":
            continue
        for binding in context.findall(f"{{{namespace}}}mousebind"):
            button = binding.get("button", "")
            if button.startswith("A-"):
                binding.set("button", "W-" + button[2:])
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    configure(Path("/etc/xdg/openbox/rc.xml"), Path.home() / ".config/openbox/rc.xml")
